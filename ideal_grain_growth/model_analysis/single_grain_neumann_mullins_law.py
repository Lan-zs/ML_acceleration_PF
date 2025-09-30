import torch.nn as nn
import random
import torch
import numpy as np
import joblib
import time
import os
import sys
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from models.residual_net import ResidualConcatNet
from utils.visualization import (
    create_bwr_custom_colormap,
    plot_grain_kinetics_comparison,
    plot_evolution_process
)


def create_grid(size=512, radius=100, n_lines=8, distance_threshold=3, sigma=2.0):
    """Create grid data."""
    # 创建坐标网格
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    X, Y = np.meshgrid(x, y)

    # 创建空白图像
    grid = np.ones((size, size))
    points = []

    # 生成圆周上的点
    theta = np.linspace(0, 2 * np.pi, 1000)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_points = np.column_stack((circle_x, circle_y))
    points.extend(circle_points.tolist())

    # 生成角平分线上的点
    angles = np.linspace(0, 2 * np.pi, n_lines, endpoint=False)

    # 对每条线进行采样
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        for t in np.linspace(radius, size / 2, 500):  # 从圆的半径开始采样
            x_pos = t * dx
            y_pos = t * dy
            points.append([x_pos, y_pos])

    points = np.array(points)

    # 为每个网格点计算到最近线条的距离
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    distances = cdist(grid_points, points)
    min_distances = distances.min(axis=1).reshape(size, size)

    # 根据距离阈值设置值
    grid[min_distances <= distance_threshold] = 0.7

    # 应用高斯平滑
    smoothed_grid = ndimage.gaussian_filter(grid, sigma=sigma, mode='wrap')

    return smoothed_grid


def save_tecplot(phi_slice, output_path, I, J):
    """Save one time-step phi slice in Tecplot format."""
    with open(output_path, 'w') as f:
        # 写入头部信息
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        # 写入数据
        for value in phi_slice.flatten():
            f.write(f"{value:.6f}\n")


def adjust_prediction_one(pred_grid):
    """Adjust predictions: snap values very close to 1 exactly to 1."""
    d = torch.abs(pred_grid - 1)
    adjusted_grid = torch.where(d < 1e-3, torch.ones_like(pred_grid), pred_grid)
    return adjusted_grid


def calculate_center_region_area(predicted_pm):
    """Compute area of the center region."""
    # 创建二值图像：将接近1的值设为True，其他设为False
    binary = np.isclose(predicted_pm, 1.0, atol=0.1)

    # 标记连通区域
    labeled_array, num_features = ndimage.label(binary)

    # 获取图像中心点
    center_y, center_x = np.array(predicted_pm.shape) // 2

    # 获取中心点所在的标签
    center_label = labeled_array[center_y, center_x]

    # 创建掩码，只保留中心区域
    center_mask = labeled_array == center_label

    # 计算区域面积（像素数量）
    area = np.sum(center_mask)

    return area


def imshow_plot(grid, t, output_dir):
    """Plot grid data with imshow."""
    plt.figure()
    plt.imshow(grid, cmap='viridis')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"imshow_{t:06d}.png"), dpi=250)
    plt.close()


def model_performance_visualization11(data_list, plot_config, title, step, output_dir):
    """Visualize model predictions."""
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]  # 确保axes是列表，即使只有一个元素

    for i, (data, label, config_key) in enumerate(data_list):
        im = axes[i].imshow(data, **plot_config[config_key])
        axes[i].tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()

    # 保存图像
    filename = f"{output_dir}/{title.replace(' ', '_')}_step_{step}.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def rollout_testing(model, n_lines, current_phi, output_dir, output_base_dir,
                    time_steps=500, step_interval=50):
    """Unified rollout testing with visualization and area computation."""
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 计算中心切片范围
    start, end = 128, 384
    center_slice = np.s_[start:end, start:end]

    s_time = time.time()

    # 先运行50步稳定
    for t in range(0, 50):
        print('t/time_steps:', t / time_steps, end="\n")

        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            predicted_phi_scaled, delta_phi = model(input_tensor)  # 残差
            predicted_phi = adjust_prediction_one(predicted_phi_scaled.squeeze())
            predicted_phi = predicted_phi.cpu().numpy()

        # Replace center region with predicted data
        result = current_phi.copy()
        result[center_slice] = predicted_phi[center_slice]
        current_phi = result

    # Determine run steps based on number of sides
    if n_lines == 3:
        step_ = 200
    elif n_lines == 4:
        step_ = 360
    else:
        step_ = 500

    area_list = []

    # Main loop and record area
    for t in range(0, step_):
        print('t/time_steps:', t / time_steps, end="\n")

        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            predicted_phi_scaled, delta_phi = model(input_tensor)  # 残差
            delta_phi = delta_phi.squeeze().cpu().numpy()
            predicted_phi = adjust_prediction_one(predicted_phi_scaled.squeeze())
            predicted_phi = predicted_phi.cpu().numpy()

        # Record area
        area_ = calculate_center_region_area(current_phi)
        area_list.append(area_)

        # Visualization
        if t % step_interval == 0:
            pred_diff = predicted_phi - current_phi.squeeze()  # Change relative to input

            plot_config = {
                "c_phi": {"vmin": np.min(current_phi[center_slice]),
                          "vmax": np.max(current_phi[center_slice]), 'cmap': 'viridis'},
                "p_phi": {"vmin": np.min(predicted_phi[center_slice]),
                          "vmax": np.max(predicted_phi[center_slice]), 'cmap': 'viridis'},
                "pdiff": {"vmin": -np.max(np.abs(pred_diff[center_slice])),
                          "vmax": np.max(np.abs(pred_diff[center_slice])),
                          'cmap': create_bwr_custom_colormap()},
                "temp": {"vmin": 870, "vmax": 880, 'cmap': 'viridis'}
            }

            data_list = [
                (predicted_phi[center_slice], f"Predicted Phi (Step {t + 1})", "p_phi"),
            ]

            model_performance_visualization11(data_list, plot_config, 'imshow_', t, output_dir)

        # Replace center region with predicted data
        result = current_phi.copy()
        result[center_slice] = predicted_phi[center_slice]
        current_phi = result

    # Save area data
    joblib.dump(area_list, os.path.join(output_base_dir, f"{n_lines}.pkl"))

    e_time = time.time()
    print()
    print(f'Total inference time: {e_time - s_time}')


def theoretical_slope(ns, theory_coef):
    """Compute theoretical slope under the Neumann-Mullins law."""
    return theory_coef * ((ns / 3) - 2)


def y_formatter(x, pos):
    """Custom y-axis tick formatter."""
    return f'{x * 5:.0f}'


def analyze_neumann_mullins(output_base_dir):
    """Analyze Neumann-Mullins law results."""
    # 定义Neumann-Mullins定律参数
    PI = 3.1415926535
    kev = 8.625e-5
    Q = 1.82
    Tmax = 1073.0
    T0 = 1023.0

    # 计算理论参数
    sigma = 1.0
    lambda2 = 6.0
    ep = (4 / PI) * np.sqrt(lambda2 * sigma / 2)
    w = 4 * sigma / lambda2

    mobility = np.exp(-Q / (kev * T0)) / (np.exp(-Q / (kev * Tmax)))
    M = PI * PI * mobility / (8 * lambda2)

    # 理论系数
    theory_coef = sigma * M * PI
    print("Theoretical coefficient:", theory_coef)

    # 读取所有pkl文件并处理数据
    data_dict = {}
    slopes = {}
    for i in range(3, 11):
        try:
            filename = os.path.join(output_base_dir, f"{i}.pkl")
            area_list = joblib.load(filename)

            area_diff = (np.array(area_list) - area_list[0]) / 100
            area_diff = np.array([x for x in area_diff if x > -25 and x < 35])

            time_points = np.arange(area_diff.shape[0])
            data_dict[i] = (time_points, area_diff)

            # 线性回归获取实际斜率
            model = LinearRegression()
            X = time_points.reshape(-1, 1)
            y = area_diff.reshape(-1, 1)
            model.fit(X, y)
            slopes[i] = model.coef_[0][0]
        except FileNotFoundError:
            continue

    if not slopes:
        print("No data files found; cannot analyze.")
        return


    ns_array = np.array(list(slopes.keys()))
    actual_slopes = np.array([slopes[ns] for ns in ns_array])

    print("Actual slopes by grain sides:", {ns: slopes[ns] for ns in ns_array})
    print("Theoretical slopes by grain sides:", {ns: theoretical_slope(ns, theory_coef) for ns in ns_array})

    # 创建图表: 面积随时间变化及拟合曲线
    plt.figure(figsize=(10, 6))

    # 理论预测的颜色映射
    ns_values = sorted(list(data_dict.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ns_values)))
    color_map = dict(zip(ns_values, colors))

    # 绘制每组数据和其线性拟合
    for idx, (ns, (time_points, area_diff)) in enumerate(sorted(data_dict.items())):
        color = color_map[ns]

        # 绘制散点
        plt.scatter(time_points[::3], area_diff[::3], marker='o', facecolors='none',
                    edgecolors=color, s=40, label=f'ns={ns}')


        theo_slope = theoretical_slope(ns, theory_coef)
        theo_pred = theo_slope * time_points
        plt.plot(time_points, theo_pred, '--', color='r', linewidth=2, alpha=0.7)

    # 自定义y轴的刻度显示
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))

    # 设置图表属性
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel(r'S(t)-S(0) ($\mu m^2$)', fontsize=20)
    plt.title('Neumann-Mullins Law: Area Evolution', fontsize=30)
    # plt.ylim(-30, 40)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'area_time_evolution.png'),
                dpi=300, bbox_inches='tight')

    # Extra plot: slope vs. sides (theory vs. simulation)
    plt.figure(figsize=(10, 6))

    # 实际数据点
    plt.scatter(ns_array, actual_slopes, color='blue', marker='o', s=100,
                label='DeepLearning Simulation')

    # 原始理论预测
    ns_theory = np.linspace(3, 10, 100)
    slope_theory_orig = [theoretical_slope(ns, theory_coef) for ns in ns_theory]
    plt.plot(ns_theory, slope_theory_orig, 'g--', linewidth=2, label='Theory')

    # 绘制n_s=6的竖线
    plt.axvline(x=6, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # 设置图表属性
    plt.xlabel('Number of Grain Sides ($n_s$)', fontsize=20)
    plt.ylabel('dA/dt', fontsize=20)
    plt.title('Neumann-Mullins Law: Theory vs Simulation', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'theory_vs_simulation.png'),
                dpi=300, bbox_inches='tight')


def load_model(model_path):
    """Load pretrained model."""
    model = ResidualConcatNet(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d,
        min_value=0.7,
        max_value=1.0
    )

    model.cuda()

    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model


def main():
    """Main function."""
    # Define paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'best_checkpoint.pt'
    )
    output_base_dir = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'single_grain_neumann_mullins_law_result'
    )

    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 加载模型
    model = load_model(model_path)

    # 检查是否所有PKL文件都已存在
    all_pkl_exist = True
    for n_lines in range(3, 11):
        if not os.path.exists(os.path.join(output_base_dir, f"{n_lines}.pkl")):
            all_pkl_exist = False
            break

    # If all PKL files exist, analyze directly
    if all_pkl_exist:
        print("All data files exist; analyzing directly")
        analyze_neumann_mullins(output_base_dir)
        return

    # Otherwise, run simulation to generate data
    for n_lines in range(3, 11):
        print(f"Processing n_lines = {n_lines}")
        output_dir = os.path.join(output_base_dir, f"polygon{n_lines}_test")

        # Skip if PKL already exists for this number of sides
        if os.path.exists(os.path.join(output_base_dir, f"{n_lines}.pkl")):
            print(f"PKL file {n_lines}.pkl exists, skipping")
            continue

        # Create grid image
        grid_data = create_grid(size=512, radius=50, n_lines=n_lines,
                                distance_threshold=2, sigma=0.5)

        # Run simulation
        rollout_testing(model, n_lines, grid_data, output_dir, output_base_dir,
                        time_steps=501, step_interval=20)

    # After generation, run analysis
    analyze_neumann_mullins(output_base_dir)


if __name__ == '__main__':
    main()