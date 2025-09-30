import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from torch.autograd import grad
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet

def create_bwr_custom_colormap():
    """创建一个自定义的红-蓝-白颜色映射，数值0为白色，-1为蓝色，1为红色"""
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    return cmap


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def compute_gradient_dual_input(model, combined_input, output_point):
    """
    计算输出点对两个输入通道的梯度

    参数:
        model: 神经网络模型
        combined_input: 组合输入张量 [batch, 2, H, W]
        output_point: 输出点坐标 (y, x)

    返回:
        两个输入通道的梯度 (phi_gradient, temp_gradient)
    """
    # 确保输入需要梯度
    combined_input = combined_input.clone().detach().requires_grad_(True)

    # 前向传播
    predicted, _ = model(combined_input)
    predicted = predicted.squeeze()

    # 选择特定的输出点
    y, x = output_point
    target_output = predicted[y, x]

    # 计算对整个combined_input的梯度
    input_gradient = grad(target_output, combined_input, retain_graph=True)[0]

    # 分离两个通道的梯度
    phi_gradient = input_gradient[0, 0]  # 第0个通道（phi）
    temp_gradient = input_gradient[0, 1]  # 第1个通道（temp）

    return phi_gradient, temp_gradient


def load_model(checkpoint_path):
    """Load model from .pt path."""
    # 判断模型类型
    if 'resnet' in checkpoint_path.lower():
        print("Loading ResNet model")
        model = ResidualConcatNet(
            in_channels=2,
            base_channels=96,
            num_blocks=10,
            kernel_size=3,
            act_fn=nn.GELU,  # GELU activation function
            norm_2d=nn.BatchNorm2d,
            min_value=0.7,
            max_value=1.0
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    return model


def load_data(h5_path, frame_idx=0):
    """Load specific frame from H5 file."""
    # Physical parameters
    Q = 1.82
    kev = 8.625e-5
    Tmax = 1073.0

    with h5py.File(h5_path, 'r') as f:
        phi_data = f['phi_data'][frame_idx]
        temp_data = f['temp_data'][:]

        # Mobility normalization of temperature
        temp_data = np.exp(-Q / (kev * temp_data)) / np.exp(-Q / (kev * Tmax))

    return phi_data, temp_data


def analyze_receptive_field_dual_channel(model, phi_image, temp_image, points, save_path, crop_region):
    """Analyze dual-channel receptive fields for specified points and save results."""
    # 准备输入图像
    phi_tensor = torch.from_numpy(phi_image).float()
    temp_tensor = torch.from_numpy(temp_image).float()

    # Create combined input
    combined_input = torch.cat((phi_tensor.unsqueeze(0).unsqueeze(0),
                                temp_tensor.unsqueeze(0).unsqueeze(0)), dim=1)

    # Move model to CPU
    model = model.cpu()
    model.eval()

    # Get output
    with torch.no_grad():
        predicted, delta = model(combined_input)

    output_np = predicted.squeeze().numpy()
    residual_np = output_np - phi_image

    # Store gradient maps
    all_phi_grad_maps = []
    all_temp_grad_maps = []

    # Compute gradients per point
    for i, point in enumerate(points):
        y, x = point
        phi_grad, temp_grad = compute_gradient_dual_input(model, combined_input, (y, x))

        # Absolute gradient
        phi_grad_magnitude = torch.abs(phi_grad).detach().numpy()
        temp_grad_magnitude = torch.abs(temp_grad).detach().numpy()

        # Normalize gradients
        if phi_grad_magnitude.max() != 0:
            phi_grad_normalized = phi_grad_magnitude / phi_grad_magnitude.max()
        else:
            phi_grad_normalized = phi_grad_magnitude

        if temp_grad_magnitude.max() != 0:
            temp_grad_normalized = temp_grad_magnitude / temp_grad_magnitude.max()
        else:
            temp_grad_normalized = temp_grad_magnitude

        all_phi_grad_maps.append(phi_grad_normalized)
        all_temp_grad_maps.append(temp_grad_normalized)

    # Combine gradients
    combined_phi_grad = np.zeros_like(phi_image)
    combined_temp_grad = np.zeros_like(temp_image)

    for phi_grad_map in all_phi_grad_maps:
        combined_phi_grad += phi_grad_map

    for temp_grad_map in all_temp_grad_maps:
        combined_temp_grad += temp_grad_map

    # Normalize combined gradient map
    if combined_phi_grad.max() != 0:
        combined_phi_grad = combined_phi_grad / combined_phi_grad.max()

    if combined_temp_grad.max() != 0:
        combined_temp_grad = combined_temp_grad / combined_temp_grad.max()

    # Save data for visualization
    data_to_save = {
        'phi_input_image': phi_image,
        'temp_input_image': temp_image,
        'output_image': output_np,
        'residual': residual_np,
        'combined_phi_gradient': combined_phi_grad,
        'combined_temp_gradient': combined_temp_grad,
        'points': points,
        'crop_region': crop_region  # y_min, y_max, x_min, x_max
    }

    # 保存数据
    np.save(os.path.join(save_path, "visualization_data.npy"), data_to_save)

    # Create visualization: first row phi, second row temperature
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Crop region
    y_min, y_max, x_min, x_max = crop_region
    image_extent = [x_min, x_max, y_max, y_min]

    # === Row 1: Phi channel ===

    # Phi input
    axes[0, 0].imshow(phi_image[y_min:y_max, x_min:x_max], cmap='viridis', extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[0, 0].plot(x, y, 'ro', markersize=6)
            axes[0, 0].text(x + 2, y, f'({x},{y})', color='white', fontsize=8,
                            verticalalignment='center', weight='bold')
    axes[0, 0].set_title('Phi Input Channel')
    axes[0, 0].set_xlabel('Original X-coordinate')
    axes[0, 0].set_ylabel('Original Y-coordinate')

    # Phi receptive field heatmap
    min_val, max_val = phi_image.min(), phi_image.max()
    if max_val > min_val:
        normalized_phi_input = (phi_image - min_val) / (max_val - min_val)
    else:
        normalized_phi_input = phi_image

    cropped_phi_input = normalized_phi_input[y_min:y_max, x_min:x_max]
    cropped_phi_grad = combined_phi_grad[y_min:y_max, x_min:x_max]

    phi_heatmap = plt.cm.jet(cropped_phi_grad)[:, :, :3]
    alpha = 0.7
    phi_blend = (1 - alpha) * np.stack([cropped_phi_input] * 3, axis=-1) + alpha * phi_heatmap
    phi_blend = np.clip(phi_blend, 0, 1)

    axes[0, 1].imshow(phi_blend, extent=image_extent)
    axes[0, 1].set_title('Phi Channel Receptive Field')
    axes[0, 1].set_xlabel('Original X-coordinate')
    axes[0, 1].tick_params(axis='y', labelleft=False)

    # Residual
    axes[0, 2].imshow(residual_np[y_min:y_max, x_min:x_max],
                      cmap=create_bwr_custom_colormap(), extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[0, 2].plot(x, y, 'ro', markersize=6)
            axes[0, 2].text(x + 2, y, f'({x},{y})', color='black', fontsize=8,
                            verticalalignment='center', weight='bold')
    axes[0, 2].set_title('Residual (Output - Phi Input)')
    axes[0, 2].set_xlabel('Original X-coordinate')
    axes[0, 2].tick_params(axis='y', labelleft=False)

    # === Row 2: Temperature channel ===

    # Temperature input
    axes[1, 0].imshow(temp_image[y_min:y_max, x_min:x_max], cmap='coolwarm', extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[1, 0].plot(x, y, 'ko', markersize=6)
            axes[1, 0].text(x + 2, y, f'({x},{y})', color='white', fontsize=8,
                            verticalalignment='center', weight='bold')
    axes[1, 0].set_title('Temperature Input Channel')
    axes[1, 0].set_xlabel('Original X-coordinate')
    axes[1, 0].set_ylabel('Original Y-coordinate')

    # Temperature receptive field heatmap
    min_val, max_val = temp_image.min(), temp_image.max()
    if max_val > min_val:
        normalized_temp_input = (temp_image - min_val) / (max_val - min_val)
    else:
        normalized_temp_input = temp_image

    cropped_temp_input = normalized_temp_input[y_min:y_max, x_min:x_max]
    cropped_temp_grad = combined_temp_grad[y_min:y_max, x_min:x_max]

    temp_heatmap = plt.cm.jet(cropped_temp_grad)[:, :, :3]
    temp_blend = (1 - alpha) * np.stack([cropped_temp_input] * 3, axis=-1) + alpha * temp_heatmap
    temp_blend = np.clip(temp_blend, 0, 1)

    axes[1, 1].imshow(temp_blend, extent=image_extent)
    axes[1, 1].set_title('Temperature Channel Receptive Field')
    axes[1, 1].set_xlabel('Original X-coordinate')
    axes[1, 1].tick_params(axis='y', labelleft=False)

    # Temperature residual panel (show temperature field itself)
    axes[1, 2].imshow(temp_image[y_min:y_max, x_min:x_max],
                      cmap='coolwarm', extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[1, 2].plot(x, y, 'ko', markersize=6)
            axes[1, 2].text(x + 2, y, f'({x},{y})', color='black', fontsize=8,
                            verticalalignment='center', weight='bold')
    axes[1, 2].set_title('Temperature Field Distribution')
    axes[1, 2].set_xlabel('Original X-coordinate')
    axes[1, 2].tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "dual_channel_receptive_field_analysis.png"), dpi=300)
    # plt.show()
    plt.close()

    return True


def main():
    # Paths (repo-relative)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    h5_file_path = os.path.join(repo_root, 'dataset', 'grain_growth_temp', 'Evaluation', 'SinHeat_P1_I256_J256_ndt100_mint2000_maxt100000_cleaned75.h5')
    checkpoint_path = os.path.join(repo_root, 'non-isothermal_grain_growth', 'experiment_results', 'augmentation ScaledResNet_C96_NB10_K3_ndt1', 'best_checkpoint.pt')
    save_dir = os.path.join(repo_root, 'non-isothermal_grain_growth', 'experiment_results', 'augmentation ScaledResNet_C96_NB10_K3_ndt1', 'specified_points_analysis')

    # Frame to analyze
    frame_idx = 100

    # Sampling region and count
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    num_points = 10

    # Crop area to display
    crop_region = (0, 100, 0, 100)  # y_min, y_max, x_min, x_max

    # Ensure save directory exists
    save_dir = save_dir + f"_frame_{frame_idx}"
    ensure_dir(save_dir)

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path)

    # Load data
    print("Loading data...")
    phi_image, temp_image = load_data(h5_file_path, frame_idx)

    # Sample points
    random.seed(64)  # for reproducibility
    # specified_points = [(random.randint(x_min, x_max), random.randint(y_min, y_max))
    #                     for _ in range(num_points)]
    # specified_points = [(60, 15), (70, 84), (56, 70), (83, 6), (26, 34),(50,40), (90, 80), (73, 30),(28,70), (10, 53)]
    # specified_points = [(60, 8), (70, 84), (56, 70), (83, 6), (26, 34),(50,40), (90, 80), (73, 30),(28,70), (10, 53)]
    specified_points = [(60, 8), (70, 84), (56, 70), (83, 6), (26, 34),(50,40), (90, 64), (73, 30),(28,80), (10, 53)]

    print(f"Sampled points: {specified_points}")

    # Analyze receptive field
    print(f"Analyzing dual-channel receptive field for frame {frame_idx} ...")
    analyze_receptive_field_dual_channel(
        model, phi_image, temp_image, specified_points, save_dir, crop_region
    )

    print(f"Analysis done! Results saved to {save_dir}")

    # Print quick stats
    print(f"\n=== Stats ===")
    print(f"Phi range: [{phi_image.min():.4f}, {phi_image.max():.4f}]")
    print(f"Temp range: [{temp_image.min():.4f}, {temp_image.max():.4f}]")
    print(f"Region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"Num points: {num_points}")


if __name__ == "__main__":
    main()