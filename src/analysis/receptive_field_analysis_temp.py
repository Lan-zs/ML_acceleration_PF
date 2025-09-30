"""
Receptive field analysis module for theoretical and effective RF of models.

Key capabilities:
1. Analyze RFs for phase-field and temperature input channels separately
2. Stratified sampling based on phase-field structural features and temperature values
3. Multi-image indexing statistics and visualizations
4. Generate detailed statistical reports and plots
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torch.autograd import grad
from matplotlib.colors import LinearSegmentedColormap
import h5py
import time
from collections import defaultdict
from scipy import ndimage
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet

def create_bwr_custom_colormap():
    """Create a blue-white-red colormap; 0 is white, -1 blue, 1 red."""
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    return cmap


def ensure_dir(directory):
    """Ensure directory exists; create if missing."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_tecplot(data, output_path, I, J):
    """
    Save data in Tecplot format.

    Args:
        data: 2D array
        output_path: output file path
        I, J: grid dimensions
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        # Write data
        for value in data.flatten():
            f.write(f"{value:.6f}\n")


def compute_theoretical_receptive_field(model):
    """
    Compute theoretical RF size for the model.

    Args:
        model: neural network model (UNet/ResNet or variants)

    Returns:
        theoretical RF size (int)
    """

    # Handle ResNet models
    if isinstance(model, ResidualConcatNet):
        # Compute theoretical RF for ResNet
        num_blocks = model.num_blocks
        kernel_size = model.res_blocks[0][0].conv[1].kernel_size[0]

        # Initial layer RF
        receptive_field_size = kernel_size

        # Each residual block increases RF
        for _ in range(num_blocks):
            # Two conv layers per residual block
            receptive_field_size += 2 * (kernel_size - 1)

        # Final 1x1 conv does not increase RF
        return receptive_field_size
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")


def calculate_effective_rf_size(grad_map, threshold=0.05):
    """
    Compute effective RF size.

    Args:
        grad_map: normalized gradient map
        threshold: threshold relative to the center max

    Returns:
        RF size in pixels
    """
    # Take max as center value
    center_val = grad_map.max()

    # Compute threshold relative to center
    thresh_val = center_val * threshold

    # Count pixels above threshold
    rf_size = np.sum(grad_map > thresh_val)

    return rf_size


def compute_gradient_dual_input(model, input_tensor, output_point, is_tem_model=False, current_temp=None):
    """
    Compute gradients of an output point with respect to both input channels.

    Args:
        model: model
        input_tensor: input tensor (phase-field and temperature)
        output_point: output coordinate (y, x)
        is_tem_model: flag for temp model
        current_temp: temperature field (for tem_model)

    Returns:
        gradients for both channels
    """
    # Ensure inputs require grad
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward
    if is_tem_model and current_temp is not None:
        current_temp = current_temp.clone().detach().requires_grad_(True)
        predicted, _ = model(input_tensor, current_temp)
    else:
        predicted, _ = model(input_tensor)

    predicted = predicted.squeeze()

    # Select location
    y, x = output_point
    target_output = predicted[y, x]

    # Compute gradients
    if is_tem_model and current_temp is not None:
        input_gradient = grad(target_output, input_tensor, retain_graph=True)[0]
        temp_gradient = grad(target_output, current_temp, retain_graph=True)[0]
        return input_gradient, temp_gradient
    else:
        input_gradient = grad(target_output, input_tensor, retain_graph=True)[0]
        return input_gradient, None


def identify_grain_regions(image):
    """
    Identify grain regions and assign discrete labels per pixel (iterative).

    Args:
        image: input image

    Returns:
        labeled image with unique identifiers per pixel
    """
    # Grain mask (values > 0.72) and initial labeling
    grain_mask = (image > 0.72).astype(int)
    labeled_image, num_features = ndimage.label(grain_mask)

    # Iterate until all pixels are labeled
    while np.any(labeled_image == 0):
        unlabeled_mask = (labeled_image == 0)
        unlabeled_indices = np.where(unlabeled_mask)

        for i in range(len(unlabeled_indices[0])):
            row, col = unlabeled_indices[0][i], unlabeled_indices[1][i]

            # Extract 8-neighborhood labels
            neighbors = labeled_image[max(0, row - 1):min(labeled_image.shape[0], row + 2),
                        max(0, col - 1):min(labeled_image.shape[1], col + 2)].flatten()

            # Remove zeros (unlabeled)
            neighbors = neighbors[neighbors != 0]

            # If any labeled neighbors exist
            if len(neighbors) > 0:
                # Pick most frequent neighbor label
                counts = np.bincount(neighbors)
                most_common_neighbor = np.argmax(counts)

                # Assign label
                labeled_image[row, col] = most_common_neighbor

    return labeled_image


def identify_triple_junctions(labeled_image):
    """
    Identify triple-junction regions.

    Args:
        labeled_image: labeled grain image

    Returns:
        boolean mask for triple junctions
    """
    h, w = labeled_image.shape
    triple_junction_mask = np.zeros((h, w), dtype=bool)

    # Check whether 3x3 neighborhood has >=3 distinct grains
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Distinct grain ids in 3x3 window
            neighborhood = labeled_image[i - 1:i + 2, j - 1:j + 2]
            unique_grains = np.unique(neighborhood)

            # Mark as triple-junction if >=3
            if len(unique_grains) >= 3:
                triple_junction_mask[i, j] = True

    return triple_junction_mask


def identify_grain_boundaries(image, triple_junction_mask, threshold_low=0.699, threshold_high=0.9):
    """
    Identify grain boundaries while excluding triple junctions.

    Args:
        image: original image
        triple_junction_mask: boolean mask of triple junctions
        threshold_low: lower threshold for GB
        threshold_high: upper threshold for GB

    Returns:
        boolean mask of grain boundaries
    """
    # GB mask (values near ~0.7)
    grain_boundary_mask = np.logical_and(
        image >= threshold_low,
        image <= threshold_high
    )

    # Exclude triple junctions
    grain_boundary_mask = np.logical_and(
        grain_boundary_mask,
        ~triple_junction_mask
    )

    return grain_boundary_mask


def identify_grain_interiors(image, triple_junction_mask, grain_boundary_mask, threshold=0.99, window_size=9):
    """
    Identify grain interiors: values near 1 with surrounding window near 1.

    Args:
        image: original image
        triple_junction_mask: triple junction mask
        grain_boundary_mask: grain boundary mask
        threshold: threshold for interiors
        window_size: window size for local check

    Returns:
        boolean mask of grain interiors
    """
    h, w = image.shape
    half_size = window_size // 2
    grain_interior_mask = np.zeros((h, w), dtype=bool)

    # Initial interior mask (values near 1)
    interior_initial = (image > threshold)

    # Check window for each candidate interior point
    for i in range(half_size, h - half_size):
        for j in range(half_size, w - half_size):
            if interior_initial[i, j]:
                window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
                # All pixels in window are near 1
                if np.all(window > threshold):
                    grain_interior_mask[i, j] = True

    # Exclude triple junction and grain boundary
    grain_interior_mask = np.logical_and(
        grain_interior_mask,
        ~np.logical_or(triple_junction_mask, grain_boundary_mask)
    )

    return grain_interior_mask


def select_stratified_points_advanced(phi_image, temp_image, num_points_per_category=5):
    """
    根据相场结构特征和温度场分层选择采样点

    参数:
        phi_image: 相场图像
        temp_image: 温度场图像
        num_points_per_category: 每类采样点数量

    返回:
        采样点字典，格式为{'category': [(y, x, phi_value, temp_value), ...]}
    """
    h, w = phi_image.shape
    margin = 10  # 边缘margin

    # 识别晶粒区域
    labeled_image = identify_grain_regions(phi_image)

    # 识别三晶交、晶界和晶内区域
    triple_junction_mask = identify_triple_junctions(labeled_image)
    grain_boundary_mask = identify_grain_boundaries(phi_image, triple_junction_mask)
    grain_interior_mask = identify_grain_interiors(phi_image, triple_junction_mask, grain_boundary_mask)

    # 温度分类的阈值
    # 物理参数
    Q = 1.82
    kev = 8.625e-5
    Tmax = 1073.0

    # 温度阈值
    temp_high_threshold = np.exp(-Q / (kev * 1023)) / np.exp(-Q / (kev * Tmax))
    temp_mid_threshold = np.exp(-Q / (kev * 973)) / np.exp(-Q / (kev * Tmax))

    # 初始化结果字典
    categories = [
        'triple_junction_temp_high', 'triple_junction_temp_mid', 'triple_junction_temp_low',
        'grain_boundary_temp_high', 'grain_boundary_temp_mid', 'grain_boundary_temp_low',
        'grain_interior_temp_high', 'grain_interior_temp_mid', 'grain_interior_temp_low'
    ]

    sampled_points = {cat: [] for cat in categories}
    masks = {}

    # 创建相场结构类别掩码
    masks['triple_junction'] = triple_junction_mask
    masks['grain_boundary'] = grain_boundary_mask
    masks['grain_interior'] = grain_interior_mask

    # 创建温度类别掩码
    masks['temp_high'] = (temp_image >= temp_high_threshold)
    masks['temp_mid'] = (temp_image >= temp_mid_threshold) & (temp_image < temp_high_threshold)
    masks['temp_low'] = (temp_image < temp_mid_threshold)

    # 对边缘进行过滤（用于所有掩码）
    for key in masks:
        masks[key][:margin, :] = False
        masks[key][-margin:, :] = False
        masks[key][:, :margin] = False
        masks[key][:, -margin:] = False

    # 生成复合掩码并采样点
    for phi_cat in ['triple_junction', 'grain_boundary', 'grain_interior']:
        for temp_cat in ['temp_high', 'temp_mid', 'temp_low']:
            cat_name = f"{phi_cat}_{temp_cat}"
            compound_mask = masks[phi_cat] & masks[temp_cat]

            # 获取符合条件的点
            points = np.argwhere(compound_mask)

            if len(points) > 0:
                # 随机选择点
                selected_indices = np.random.choice(
                    len(points),
                    min(num_points_per_category, len(points)),
                    replace=False
                )

                for idx in selected_indices:
                    y, x = points[idx]
                    sampled_points[cat_name].append((y, x, phi_image[y, x], temp_image[y, x]))
            else:
                print(f"警告: 类别 {cat_name} 没有找到符合条件的点")

    # 额外添加相场结构分类（不考虑温度）
    for phi_cat in ['triple_junction', 'grain_boundary', 'grain_interior']:
        sampled_points[phi_cat] = []
        for temp_cat in ['temp_high', 'temp_mid', 'temp_low']:
            cat_name = f"{phi_cat}_{temp_cat}"
            sampled_points[phi_cat].extend(sampled_points[cat_name])
            if len(sampled_points[phi_cat]) >= num_points_per_category:
                sampled_points[phi_cat] = sampled_points[phi_cat][:num_points_per_category]
                break

    # 额外添加温度场分类（不考虑相场结构）
    for temp_cat in ['temp_high', 'temp_mid', 'temp_low']:
        sampled_points[temp_cat] = []
        for phi_cat in ['triple_junction', 'grain_boundary', 'grain_interior']:
            cat_name = f"{phi_cat}_{temp_cat}"
            sampled_points[temp_cat].extend(sampled_points[cat_name])
            if len(sampled_points[temp_cat]) >= num_points_per_category:
                sampled_points[temp_cat] = sampled_points[temp_cat][:num_points_per_category]
                break

    return sampled_points


def visualize_receptive_field_advanced(model, phi_image, temp_image, sampled_points, save_dir, idx=0):
    """
    可视化模型的感受野，基于高级分层采样

    参数:
        model: 神经网络模型
        phi_image: 相场输入图像
        temp_image: 温度场输入图像
        sampled_points: 分层采样的点的字典
        save_dir: 保存结果的目录
        idx: 图像索引

    返回:
        grad_maps: 梯度图字典，每个类别包含一个元组列表[(phi_grad, temp_grad), ...]
        points_data: 采样点坐标字典，每个类别包含一个元组列表[(y, x), ...]
    """
    # 创建保存目录
    save_path = os.path.join(save_dir, f"image_{idx}")
    ensure_dir(save_path)

    # 准备输入图像
    phi_tensor = torch.from_numpy(phi_image).float()
    temp_tensor = torch.from_numpy(temp_image).float()

    # 创建组合输入
    combined_input = torch.cat((phi_tensor.unsqueeze(0).unsqueeze(0),
                                temp_tensor.unsqueeze(0).unsqueeze(0)), dim=1)

    # 将模型转移到CPU
    model = model.cpu()
    model.eval()

    # 获取输出
    with torch.no_grad():
        predicted, delta = model(combined_input)

    predicted_np = predicted.squeeze().numpy()
    delta_np = predicted_np - phi_image

    # 分析每个类别的采样点
    grad_maps = {}
    points_data = {}

    for category, points in sampled_points.items():
        print(category)
        grad_maps[category] = []
        points_data[category] = []

        for i, (y, x, phi_val, temp_val) in tqdm(
                enumerate(points),
                total=len(points),
                desc="Processing",
                unit="point",
                ncols=100  # 进度条宽度
        ):
            # print(i)
            # 计算梯度
            grad_tensor, _ = compute_gradient_dual_input(model, combined_input, (y, x))

            # 提取两个通道的梯度
            phi_input_grad = grad_tensor[0, 0].detach().numpy()
            temp_input_grad = grad_tensor[0, 1].detach().numpy()

            # 取梯度的绝对值并归一化
            phi_grad_magnitude = np.abs(phi_input_grad)
            temp_grad_magnitude = np.abs(temp_input_grad)

            phi_grad_normalized = phi_grad_magnitude / (phi_grad_magnitude.max() + 1e-10)
            temp_grad_normalized = temp_grad_magnitude / (temp_grad_magnitude.max() + 1e-10)

            # 存储梯度和点信息
            grad_maps[category].append((phi_grad_normalized, temp_grad_normalized))
            points_data[category].append((y, x))

    return grad_maps, points_data


def extract_and_center_rf(grad_normalized, y, x, rf_radius, h, w, theoretical_rf):
    """
    提取感受野区域并居中放置

    参数:
        grad_normalized: 归一化的梯度图
        y, x: 中心点坐标
        rf_radius: 理论感受野半径
        h, w: 输入图像高度和宽度
        theoretical_rf: 理论感受野大小

    返回:
        居中放置的感受野区域
    """
    # 提取感受野区域
    y_min = max(0, y - rf_radius)
    y_max = min(h, y + rf_radius + 1)
    x_min = max(0, x - rf_radius)
    x_max = min(w, x + rf_radius + 1)

    rf_region = grad_normalized[y_min:y_max, x_min:x_max]

    # 将感受野区域放置在理论感受野中心
    rf_centered = np.zeros((theoretical_rf, theoretical_rf))

    # 计算偏移量
    y_offset = rf_radius - (y - y_min)
    x_offset = rf_radius - (x - x_min)

    # 放置感受野区域
    y_rf_min = max(0, y_offset)
    y_rf_max = min(theoretical_rf, y_offset + rf_region.shape[0])
    x_rf_min = max(0, x_offset)
    x_rf_max = min(theoretical_rf, x_offset + rf_region.shape[1])

    y_reg_min = max(0, -y_offset)
    y_reg_max = y_reg_min + (y_rf_max - y_rf_min)
    x_reg_min = max(0, -x_offset)
    x_reg_max = x_reg_min + (x_rf_max - x_rf_min)

    rf_centered[y_rf_min:y_rf_max, x_rf_min:x_rf_max] = rf_region[y_reg_min:y_reg_max, x_reg_min:x_reg_max]

    return rf_centered


def analyze_category_receptive_field(model, grad_maps, points, category, save_path, input_shape, threshold=0.01):
    """
    分析特定类别点的感受野，分别计算相场和温度场通道

    参数:
        model: 神经网络模型
        grad_maps: 类别的梯度图列表 [(phi_grad, temp_grad), ...]
        points: 类别的点坐标列表 [(y, x), ...]
        category: 类别名称
        save_path: 保存目录
        input_shape: 输入图像形状 (h, w)
        threshold: 阈值，用于计算有效感受野

    返回:
        结果元组: (
            phi_rf_stats: (直径, 理论RF, 比例, 像素数),
            temp_rf_stats: (直径, 理论RF, 比例, 像素数)
        )
    """
    if not grad_maps or not points:
        print(f"警告: {category} 类别没有采样点，跳过分析")
        return None, None

    # 计算理论感受野
    theoretical_rf = compute_theoretical_receptive_field(model)
    rf_radius = theoretical_rf // 2
    h, w = input_shape

    # 准备累积梯度，分开相场和温度场两个通道
    accumulated_phi_rf = np.zeros((theoretical_rf, theoretical_rf))
    accumulated_temp_rf = np.zeros((theoretical_rf, theoretical_rf))
    phi_rf_sizes = []
    temp_rf_sizes = []

    # 处理每个采样点
    for i, ((phi_grad, temp_grad), (y, x)) in enumerate(zip(grad_maps, points)):
        # 提取并居中感受野，分别处理两个通道
        phi_rf_centered = extract_and_center_rf(phi_grad, y, x, rf_radius, h, w, theoretical_rf)
        temp_rf_centered = extract_and_center_rf(temp_grad, y, x, rf_radius, h, w, theoretical_rf)

        # 累积感受野
        accumulated_phi_rf += phi_rf_centered
        accumulated_temp_rf += temp_rf_centered

        # 计算有效感受野大小，分别计算两个通道
        phi_rf_size = calculate_effective_rf_size(phi_rf_centered, threshold)
        temp_rf_size = calculate_effective_rf_size(temp_rf_centered, threshold)

        phi_rf_sizes.append(phi_rf_size)
        temp_rf_sizes.append(temp_rf_size)

    # 平均感受野
    avg_phi_rf = accumulated_phi_rf / len(points)
    avg_temp_rf = accumulated_temp_rf / len(points)

    # 归一化
    avg_phi_rf_normalized = avg_phi_rf / (avg_phi_rf.max() + 1e-10)
    avg_temp_rf_normalized = avg_temp_rf / (avg_temp_rf.max() + 1e-10)

    # 计算平均有效感受野大小（像素数）
    avg_rf_size_phi_pixels = np.mean(phi_rf_sizes)
    avg_rf_size_temp_pixels = np.mean(temp_rf_sizes)

    # 计算有效感受野直径
    avg_rf_diameter_phi = np.sqrt(avg_rf_size_phi_pixels / np.pi) * 2
    avg_rf_diameter_temp = np.sqrt(avg_rf_size_temp_pixels / np.pi) * 2

    # 计算有效感受野占理论感受野的比例
    ratio_phi = (avg_rf_diameter_phi / theoretical_rf) * 100
    ratio_temp = (avg_rf_diameter_temp / theoretical_rf) * 100

    # 可视化平均感受野，分别显示两个通道
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 相场感受野
    im1 = axes[0].imshow(avg_phi_rf_normalized, cmap='jet')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"Phi Channel\nEffective RF Diameter: {avg_rf_diameter_phi:.2f} pixels\n"
                      f"RF Pixels: {avg_rf_size_phi_pixels:.2f}, RF Ratio: {ratio_phi:.2f}%")

    # 温度场感受野
    im2 = axes[1].imshow(avg_temp_rf_normalized, cmap='jet')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Temperature Channel\nEffective RF Diameter: {avg_rf_diameter_temp:.2f} pixels\n"
                      f"RF Pixels: {avg_rf_size_temp_pixels:.2f}, RF Ratio: {ratio_temp:.2f}%")

    plt.suptitle(f"{category} Receptive Field Analysis\n"
                 f"Theoretical RF: {theoretical_rf}x{theoretical_rf}",
                 fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{category}_avg_receptive_field.png"), dpi=300)
    plt.close()

    # 相场通道的统计结果
    phi_rf_stats = (avg_rf_diameter_phi, theoretical_rf, ratio_phi, avg_rf_size_phi_pixels)

    # 温度场通道的统计结果
    temp_rf_stats = (avg_rf_diameter_temp, theoretical_rf, ratio_temp, avg_rf_size_temp_pixels)

    return phi_rf_stats, temp_rf_stats


def analyze_image_receptive_fields(model, phi_image, temp_image, image_idx, num_points_per_category, save_dir):
    """
    分析单个图像的感受野

    参数:
        model: 神经网络模型
        phi_image: 相场图像
        temp_image: 温度场图像
        image_idx: 图像索引
        save_dir: 保存目录

    返回:
        (统计结果, 梯度图, 采样点)
    """
    # 创建保存目录
    save_path = os.path.join(save_dir, f"image_{image_idx}")
    ensure_dir(save_path)

    # 进行分层采样
    sampled_points = select_stratified_points_advanced(phi_image, temp_image, num_points_per_category)  # 可视化的样本数

    # 分析并可视化感受野
    grad_maps, points_data = visualize_receptive_field_advanced(
        model, phi_image, temp_image, sampled_points, save_dir, image_idx
    )

    # 对每个类别分析感受野
    rf_stats = {'phi': {}, 'temp': {}}

    # 简化后的类别分组（相场结构分组）
    simplified_categories = {
        'triple_junction': ['triple_junction_temp_high', 'triple_junction_temp_mid', 'triple_junction_temp_low'],
        'grain_boundary': ['grain_boundary_temp_high', 'grain_boundary_temp_mid', 'grain_boundary_temp_low'],
        'grain_interior': ['grain_interior_temp_high', 'grain_interior_temp_mid', 'grain_interior_temp_low']
    }

    # 温度场分组
    temp_categories = {
        'temp_high': ['triple_junction_temp_high', 'grain_boundary_temp_high', 'grain_interior_temp_high'],
        'temp_mid': ['triple_junction_temp_mid', 'grain_boundary_temp_mid', 'grain_interior_temp_mid'],
        'temp_low': ['triple_junction_temp_low', 'grain_boundary_temp_low', 'grain_interior_temp_low']
    }

    # 所有原始类别
    original_categories = list(grad_maps.keys())

    # 分析各个原始类别
    for category in original_categories:
        if grad_maps[category] and points_data[category]:
            phi_result, temp_result = analyze_category_receptive_field(
                model, grad_maps[category], points_data[category],
                category, save_path, phi_image.shape, threshold=0.01
            )
            if phi_result and temp_result:
                rf_stats['phi'][category] = phi_result
                rf_stats['temp'][category] = temp_result

    # 合并相场结构类别
    for simplified_cat, subcats in simplified_categories.items():
        combined_grads = []
        combined_points = []

        for subcat in subcats:
            if subcat in grad_maps and grad_maps[subcat]:
                combined_grads.extend(grad_maps[subcat])
                combined_points.extend(points_data[subcat])

        if combined_grads and combined_points:
            phi_result, temp_result = analyze_category_receptive_field(
                model, combined_grads, combined_points,
                simplified_cat, save_path, phi_image.shape, threshold=0.01
            )
            if phi_result and temp_result:
                rf_stats['phi'][simplified_cat] = phi_result
                rf_stats['temp'][simplified_cat] = temp_result

    # 合并温度场类别
    for temp_cat, subcats in temp_categories.items():
        combined_grads = []
        combined_points = []

        for subcat in subcats:
            if subcat in grad_maps and grad_maps[subcat]:
                combined_grads.extend(grad_maps[subcat])
                combined_points.extend(points_data[subcat])

        if combined_grads and combined_points:
            phi_result, temp_result = analyze_category_receptive_field(
                model, combined_grads, combined_points,
                temp_cat, save_path, phi_image.shape, threshold=0.01
            )
            if phi_result and temp_result:
                rf_stats['phi'][temp_cat] = phi_result
                rf_stats['temp'][temp_cat] = temp_result

    # 所有类别合并分析
    all_grads = []
    all_points = []

    for category in original_categories:
        if category in grad_maps and grad_maps[category]:
            all_grads.extend(grad_maps[category])
            all_points.extend(points_data[category])

    if all_grads and all_points:
        phi_result, temp_result = analyze_category_receptive_field(
            model, all_grads, all_points,
            "All_Categories_Combined", save_path, phi_image.shape, threshold=0.01
        )
        if phi_result and temp_result:
            rf_stats['phi']["All_Categories_Combined"] = phi_result
            rf_stats['temp']["All_Categories_Combined"] = temp_result

    # 保存统计结果，分别记录相场和温度场通道的数据
    with open(os.path.join(save_path, "rf_statistics.txt"), "w") as f:
        f.write(f"图像 {image_idx} 的感受野统计结果:\n")
        f.write("===========================================\n\n")

        channels = ['phi', 'temp']
        channel_names = {'phi': '相场通道', 'temp': '温度场通道'}

        for channel in channels:
            f.write(f"{channel_names[channel]}统计结果:\n")
            f.write("====================\n\n")

            # 原始9类结果
            f.write("原始9类采样点结果:\n")
            f.write("--------------------\n")
            for category in original_categories:
                if category in rf_stats[channel]:
                    diameter, theo_rf, ratio, pixels = rf_stats[channel][category]
                    f.write(f"{category}类点的统计结果:\n")
                    f.write(f"  理论感受野大小: {theo_rf}x{theo_rf}\n")
                    f.write(f"  有效感受野直径: {diameter:.2f}\n")
                    f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                    f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                else:
                    f.write(f"{category}类点: 缺失\n\n")

            # 相场结构分组结果
            f.write("\n相场结构分组结果:\n")
            f.write("--------------------\n")
            for category in ['triple_junction', 'grain_boundary', 'grain_interior']:
                if category in rf_stats[channel]:
                    diameter, theo_rf, ratio, pixels = rf_stats[channel][category]
                    f.write(f"{category}类点的统计结果:\n")
                    f.write(f"  理论感受野大小: {theo_rf}x{theo_rf}\n")
                    f.write(f"  有效感受野直径: {diameter:.2f}\n")
                    f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                    f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                else:
                    f.write(f"{category}类点: 缺失\n\n")

            # 温度场分组结果
            f.write("\n温度场分组结果:\n")
            f.write("--------------------\n")
            for category in ['temp_high', 'temp_mid', 'temp_low']:
                if category in rf_stats[channel]:
                    diameter, theo_rf, ratio, pixels = rf_stats[channel][category]
                    f.write(f"{category}类点的统计结果:\n")
                    f.write(f"  理论感受野大小: {theo_rf}x{theo_rf}\n")
                    f.write(f"  有效感受野直径: {diameter:.2f}\n")
                    f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                    f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                else:
                    f.write(f"{category}类点: 缺失\n\n")

            # 所有类别合并结果
            f.write("\n所有类别合并结果:\n")
            f.write("--------------------\n")
            if "All_Categories_Combined" in rf_stats[channel]:
                diameter, theo_rf, ratio, pixels = rf_stats[channel]["All_Categories_Combined"]
                f.write(f"All_Categories_Combined类点的统计结果:\n")
                f.write(f"  理论感受野大小: {theo_rf}x{theo_rf}\n")
                f.write(f"  有效感受野直径: {diameter:.2f}\n")
                f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
            else:
                f.write("All_Categories_Combined类点: 缺失\n\n")

            f.write("\n" + "=" * 40 + "\n\n")  # 分隔线

    return rf_stats, grad_maps, points_data


def analyze_multiple_h5_files(model, model_evaluation_H5_folder, test_image_indices, num_points_per_category, output_base_dir):
    """
    分析多个H5文件中指定索引的图像

    参数:
        model: 神经网络模型
        model_evaluation_H5_folder: 包含H5文件的文件夹路径
        test_image_indices: 要分析的图像索引列表
        output_base_dir: 输出结果的基础目录

    返回:
        汇总结果字典，分别包含相场和温度场两个通道的数据
    """
    # 确保输出目录存在
    ensure_dir(output_base_dir)

    # 初始化结果
    results = {'phi': {}, 'temp': {}}
    all_image_stats = {}

    # 处理H5文件夹中的每个文件
    for h5_file in os.listdir(model_evaluation_H5_folder):
        if h5_file.endswith(".h5"):
            # 获取H5文件名前缀用于输出目录命名
            prefix = h5_file.split('_')[0]
            file_output_dir = os.path.join(output_base_dir, f"{prefix}_rf_analysis")
            ensure_dir(file_output_dir)

            h5_file_path = os.path.join(model_evaluation_H5_folder, h5_file)
            print(f"正在处理: {h5_file_path}")
            print(f"输出目录: {file_output_dir}")

            # 物理参数（用于温度场处理）
            Q = 1.82
            kev = 8.625e-5
            Tmax = 1073.0

            # 读取H5文件数据
            try:
                with h5py.File(h5_file_path, 'r') as f:
                    phi_data = f['phi_data'][:]
                    temp_data = f['temp_data'][:]
                    temp_data = np.exp(-Q / (kev * temp_data)) / np.exp(-Q / (kev * Tmax))

                    # 存储每个图像索引的统计结果
                    image_stats = {}

                    # 对每个索引进行分析
                    for idx in test_image_indices:
                        if idx < len(phi_data):
                            print(f"\n正在分析 {h5_file} 中索引 {idx} 的图像...")

                            # 获取测试图像
                            test_phi_image = phi_data[idx]
                            test_temp_image = temp_data

                            # 分析感受野
                            rf_stats, _, _ = analyze_image_receptive_fields(
                                model, test_phi_image, test_temp_image, idx, num_points_per_category, file_output_dir
                            )

                            # 存储结果
                            image_stats[idx] = rf_stats
                        else:
                            print(f"警告: 索引 {idx} 超出文件 {h5_file} 的范围，跳过此索引")

                    # 存储该文件的统计结果
                    all_image_stats[h5_file] = image_stats

                    # 汇总该文件的所有图像结果，分别对相场和温度场通道
                    file_summary_phi = aggregate_file_results(image_stats, 'phi')
                    file_summary_temp = aggregate_file_results(image_stats, 'temp')

                    results['phi'][h5_file] = file_summary_phi
                    results['temp'][h5_file] = file_summary_temp

                    # 保存该文件的汇总结果
                    save_file_summary(file_summary_phi, file_summary_temp, file_output_dir, h5_file)

            except Exception as e:
                print(f"处理文件 {h5_file} 时出错: {e}")

    # 创建所有文件的汇总结果，分别处理相场和温度场通道
    overall_summary_phi = aggregate_all_results(results['phi'])
    overall_summary_temp = aggregate_all_results(results['temp'])

    save_overall_summary(overall_summary_phi, overall_summary_temp, output_base_dir)

    return results


def aggregate_file_results(image_stats, channel_key):
    """
    汇总单个文件的多个图像结果

    参数:
        image_stats: 图像统计结果字典
        channel_key: 通道键名 ('phi' 或 'temp')

    返回:
        汇总结果
    """
    # 初始化汇总统计
    aggregated = {}

    # 收集所有类别
    all_categories = set()
    for idx, stats in image_stats.items():
        if channel_key in stats:
            all_categories.update(stats[channel_key].keys())

    # 对每个类别汇总结果
    for category in all_categories:
        diameters = []
        theoretical_rfs = []
        ratios = []
        pixels = []

        # 收集每张图像的数据
        for idx, stats in image_stats.items():
            if channel_key in stats and category in stats[channel_key] and stats[channel_key][category]:
                diameter, theo_rf, ratio, pixel_count = stats[channel_key][category]
                diameters.append(diameter)
                theoretical_rfs.append(theo_rf)
                ratios.append(ratio)
                pixels.append(pixel_count)

        # 如果有足够的数据，计算平均值
        if diameters:
            avg_diameter = np.mean(diameters)
            avg_theo_rf = np.mean(theoretical_rfs)  # 理论上所有图像应该一样
            avg_ratio = np.mean(ratios)
            avg_pixels = np.mean(pixels)

            aggregated[category] = (avg_diameter, avg_theo_rf, avg_ratio, avg_pixels)

    return aggregated


def aggregate_all_results(results):
    """
    汇总所有文件的结果

    参数:
        results: 所有文件的结果字典

    返回:
        总体汇总结果
    """
    # 初始化总体统计
    overall_stats = {}

    # 收集所有类别
    all_categories = set()
    for file_name, file_stats in results.items():
        all_categories.update(file_stats.keys())

    # 对每个类别汇总结果
    for category in all_categories:
        diameters = []
        theoretical_rfs = []
        ratios = []
        pixels = []

        # 收集每个文件的数据
        for file_name, file_stats in results.items():
            if category in file_stats and file_stats[category]:
                diameter, theo_rf, ratio, pixel_count = file_stats[category]
                diameters.append(diameter)
                theoretical_rfs.append(theo_rf)
                ratios.append(ratio)
                pixels.append(pixel_count)

        # 如果有足够的数据，计算平均值
        if diameters:
            avg_diameter = np.mean(diameters)
            avg_theo_rf = np.mean(theoretical_rfs)
            avg_ratio = np.mean(ratios)
            avg_pixels = np.mean(pixels)

            overall_stats[category] = (avg_diameter, avg_theo_rf, avg_ratio, avg_pixels)

    return overall_stats


def save_file_summary(file_summary_phi, file_summary_temp, output_dir, file_name):
    """
    保存单个文件的汇总结果

    参数:
        file_summary_phi: 相场通道的文件汇总结果
        file_summary_temp: 温度场通道的文件汇总结果
        output_dir: 输出目录
        file_name: 文件名
    """
    with open(os.path.join(output_dir, f"{file_name}_summary.txt"), "w") as f:
        f.write(f"文件 {file_name} 的感受野统计汇总结果:\n")
        f.write("==============================================\n\n")

        # 分别写入相场和温度场通道的结果
        for channel_name, summary in [("相场通道", file_summary_phi), ("温度场通道", file_summary_temp)]:
            f.write(f"\n{channel_name}统计结果:\n")
            f.write("====================\n\n")

            # 分组写入结果
            # 1. 原始9类别
            orig_cats = [cat for cat in summary.keys()
                         if any(x in cat for x in ["triple_junction_", "grain_boundary_", "grain_interior_"])]
            if orig_cats:
                f.write("原始9类采样点结果:\n")
                f.write("--------------------\n")
                for cat in sorted(orig_cats):
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 2. 相场结构分组
            phi_cats = ['triple_junction', 'grain_boundary', 'grain_interior']
            if any(cat in summary for cat in phi_cats):
                f.write("\n相场结构分组结果:\n")
                f.write("--------------------\n")
                for cat in phi_cats:
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 3. 温度场分组
            temp_cats = ['temp_high', 'temp_mid', 'temp_low']
            if any(cat in summary for cat in temp_cats):
                f.write("\n温度场分组结果:\n")
                f.write("--------------------\n")
                for cat in temp_cats:
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 4. 所有类别合并
            if "All_Categories_Combined" in summary:
                f.write("\n所有类别合并结果:\n")
                f.write("--------------------\n")
                diameter, theo_rf, ratio, pixels = summary["All_Categories_Combined"]
                f.write(f"All_Categories_Combined类点的统计结果:\n")
                f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                f.write(f"  有效感受野直径: {diameter:.2f}\n")
                f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")

            f.write("\n" + "=" * 40 + "\n")  # 分隔线


def save_overall_summary(overall_summary_phi, overall_summary_temp, output_dir):
    """
    保存总体汇总结果

    参数:
        overall_summary_phi: 相场通道的总体汇总结果
        overall_summary_temp: 温度场通道的总体汇总结果
        output_dir: 输出目录
    """
    with open(os.path.join(output_dir, "overall_rf_statistics.txt"), "w") as f:
        f.write("所有文件和图像的感受野统计汇总结果:\n")
        f.write("=======================================\n\n")

        # 分别写入相场和温度场通道的结果
        for channel_name, summary in [("相场通道", overall_summary_phi), ("温度场通道", overall_summary_temp)]:
            f.write(f"\n{channel_name}统计结果:\n")
            f.write("====================\n\n")

            # 分组写入结果
            # 1. 原始9类别
            orig_cats = [cat for cat in summary.keys()
                         if any(x in cat for x in ["triple_junction_", "grain_boundary_", "grain_interior_"])]
            if orig_cats:
                f.write("原始9类采样点结果:\n")
                f.write("--------------------\n")
                for cat in sorted(orig_cats):
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 2. 相场结构分组
            phi_cats = ['triple_junction', 'grain_boundary', 'grain_interior']
            if any(cat in summary for cat in phi_cats):
                f.write("\n相场结构分组结果:\n")
                f.write("--------------------\n")
                for cat in phi_cats:
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 3. 温度场分组
            temp_cats = ['temp_high', 'temp_mid', 'temp_low']
            if any(cat in summary for cat in temp_cats):
                f.write("\n温度场分组结果:\n")
                f.write("--------------------\n")
                for cat in temp_cats:
                    if cat in summary:
                        diameter, theo_rf, ratio, pixels = summary[cat]
                        f.write(f"{cat}类点的统计结果:\n")
                        f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                        f.write(f"  有效感受野直径: {diameter:.2f}\n")
                        f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                        f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")
                    else:
                        f.write(f"{cat}类点: 缺失\n\n")

            # 4. 所有类别合并
            if "All_Categories_Combined" in summary:
                f.write("\n所有类别合并结果:\n")
                f.write("--------------------\n")
                diameter, theo_rf, ratio, pixels = summary["All_Categories_Combined"]
                f.write(f"All_Categories_Combined类点的统计结果:\n")
                f.write(f"  理论感受野大小: {theo_rf:.0f}x{theo_rf:.0f}\n")
                f.write(f"  有效感受野直径: {diameter:.2f}\n")
                f.write(f"  有效感受野像素数: {pixels:.2f}\n")
                f.write(f"  有效/理论感受野比例: {ratio:.2f}%\n\n")

            f.write("\n" + "=" * 40 + "\n")  # 分隔线


def analyze_model_receptive_field(phi_model, model_evaluation_H5_folder, result_dir, test_image_indices=None, num_points_per_category=30):
    """
    分析模型的感受野，支持多个图像索引分析和按相场结构特征、温度场分层采样
    分别记录相场通道和温度场通道的感受野统计数据

    参数:
        phi_model: 相场模型
        model_evaluation_H5_folder: 模型评估H5文件夹路径
        result_dir: 结果保存目录
        test_image_indices: 要分析的测试图像索引列表，默认为[100, 500, 1000]
    """
    # 设置默认图像索引
    if test_image_indices is None:
        test_image_indices = [100, 500, 1000]

    # 设置输出目录
    output_base_dir = os.path.join(result_dir, "receptive_field_analysis")
    ensure_dir(output_base_dir)

    # 将模型移至CPU
    phi_model = phi_model.cpu()

    # 分析多个H5文件
    results = analyze_multiple_h5_files(
        phi_model,
        model_evaluation_H5_folder,
        test_image_indices,
        num_points_per_category,
        output_base_dir
    )

    # 打印汇总结果
    print("\n===== Receptive Field Analysis Summary =====")

    # 分别打印相场和温度场通道的结果
    channels = {'phi': '相场通道', 'temp': '温度场通道'}

    for channel_key, channel_name in channels.items():
        print(f"\n{channel_name}感受野分析结果:")
        print("=" * 40)

        for file_name, file_stats in results[channel_key].items():
            print(f"\nFile: {file_name}")

            # 按相场结构分组打印结果
            phi_categories = ['triple_junction', 'grain_boundary', 'grain_interior']
            print("  相场结构分类:")
            for cat in phi_categories:
                if cat in file_stats:
                    diameter, theo_rf, ratio, pixels = file_stats[cat]
                    print(f"    {cat}:")
                    print(f"      有效感受野直径: {diameter:.2f}")
                    print(f"      理论感受野: {theo_rf:.0f}x{theo_rf:.0f}")
                    print(f"      有效/理论感受野比例: {ratio:.2f}%")
                    print(f"      有效感受野像素数: {pixels:.2f}")
                else:
                    print(f"    {cat}: 数据缺失")

            # 按温度场分组打印结果
            temp_categories = ['temp_high', 'temp_mid', 'temp_low']
            print("\n  温度场分类:")
            for cat in temp_categories:
                if cat in file_stats:
                    diameter, theo_rf, ratio, pixels = file_stats[cat]
                    print(f"    {cat}:")
                    print(f"      有效感受野直径: {diameter:.2f}")
                    print(f"      理论感受野: {theo_rf:.0f}x{theo_rf:.0f}")
                    print(f"      有效/理论感受野比例: {ratio:.2f}%")
                    print(f"      有效感受野像素数: {pixels:.2f}")
                else:
                    print(f"    {cat}: 数据缺失")

            # 所有类别合并结果
            if "All_Categories_Combined" in file_stats:
                diameter, theo_rf, ratio, pixels = file_stats["All_Categories_Combined"]
                print("\n  所有类别合并:")
                print(f"    有效感受野直径: {diameter:.2f}")
                print(f"    理论感受野: {theo_rf:.0f}x{theo_rf:.0f}")
                print(f"    有效/理论感受野比例: {ratio:.2f}%")
                print(f"    有效感受野像素数: {pixels:.2f}")
            else:
                print("\n  所有类别合并: 数据缺失")

    return results


def analyze_single_image_receptive_field(phi_model, h5_file_path, result_dir, image_idx=100, num_points_per_category=10):
    """
    分析单个图像的感受野（用于快速分析单个图像）
    分别记录相场通道和温度场通道的感受野统计数据

    参数:
        phi_model: 相场模型
        h5_file_path: 单个H5文件的路径
        result_dir: 结果保存目录
        image_idx: 要分析的图像索引
    """
    import h5py

    # 设置输出目录
    output_dir = os.path.join(result_dir, "single_image_rf_analysis")
    ensure_dir(output_dir)

    # 将模型移至CPU
    phi_model = phi_model.cpu()
    phi_model.eval()

    # 物理参数
    Q = 1.82
    kev = 8.625e-5
    Tmax = 1073.0

    # 读取H5文件数据
    try:
        with h5py.File(h5_file_path, 'r') as f:
            phi_data = f['phi_data'][:]
            temp_data = f['temp_data'][:]
            temp_data = np.exp(-Q / (kev * temp_data)) / np.exp(-Q / (kev * Tmax))

            # 确保索引在有效范围内
            if image_idx < len(phi_data):
                test_phi_image = phi_data[image_idx]
                test_temp_image = temp_data

                # 分析感受野
                rf_stats, _, _ = analyze_image_receptive_fields(
                    phi_model, test_phi_image, test_temp_image, image_idx, num_points_per_category, output_dir
                )

                # 打印结果
                print(f"\n===== 图像 {image_idx} 的感受野分析结果 =====")

                # 分别打印相场和温度场通道的结果
                channels = {'phi': '相场通道', 'temp': '温度场通道'}

                for channel_key, channel_name in channels.items():
                    print(f"\n{channel_name}感受野分析结果:")
                    print("=" * 40)

                    # 按相场结构分组打印结果
                    phi_categories = ['triple_junction', 'grain_boundary', 'grain_interior']
                    print("相场结构分类的结果:")
                    for cat in phi_categories:
                        if cat in rf_stats[channel_key]:
                            diameter, theo_rf, ratio, pixels = rf_stats[channel_key][cat]
                            print(f"  {cat}:")
                            print(f"    有效感受野直径: {diameter:.2f}")
                            print(f"    理论感受野: {theo_rf:.0f}x{theo_rf:.0f}")
                            print(f"    有效/理论感受野比例: {ratio:.2f}%")
                            print(f"    有效感受野像素数: {pixels:.2f}")
                        else:
                            print(f"  {cat}: 数据缺失")

                    # 所有类别合并结果
                    if "All_Categories_Combined" in rf_stats[channel_key]:
                        diameter, theo_rf, ratio, pixels = rf_stats[channel_key]["All_Categories_Combined"]
                        print("\n所有类别合并结果:")
                        print(f"  有效感受野直径: {diameter:.2f}")
                        print(f"  理论感受野: {theo_rf:.0f}x{theo_rf:.0f}")
                        print(f"  有效/理论感受野比例: {ratio:.2f}%")
                        print(f"  有效感受野像素数: {pixels:.2f}")
                    else:
                        print("\n所有类别合并结果: 数据缺失")

                return rf_stats
            else:
                print(f"错误: 索引 {image_idx} 超出文件中的数据范围")

    except Exception as e:
        print(f"处理文件时出错: {e}")