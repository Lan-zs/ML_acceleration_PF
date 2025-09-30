"""
Time-varying temperature field rollout inference for grain growth model.

This script performs long-term rollout predictions with dynamic temperature fields
to evaluate the trained model's performance under non-isothermal conditions.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import gaussian_filter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet


# ============================================================================
# Utility Functions
# ============================================================================

def create_bwr_custom_colormap():
    """Create a blue-white-red custom colormap."""
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝、白、红
    return LinearSegmentedColormap.from_list('bwr_custom', colors, N=256)


def save_tecplot(data, filename, ni, nj):
    """Save data file in Tecplot format."""
    with open(filename, 'w') as f:
        f.write('TITLE = "Temperature/Phase Field Data"\n')
        f.write('VARIABLES = "X", "Y", "Value"\n')
        f.write(f'ZONE I={ni}, J={nj}, F=POINT\n')

        dx, dy = 1.0 / ni, 1.0 / nj
        for j in range(nj):
            for i in range(ni):
                y = j * dy
                x = i * dx
                value = data[j, i]
                f.write(f"{x:.6f} {y:.6f} {value:.6f}\n")


def compare_and_adjust(predicted_tensor, current_tensor, threshold=1e-2):
    """Keep original values where change magnitude is below threshold."""
    # 确保两个tensor都在同一设备上
    if predicted_tensor.device != current_tensor.device:
        current_tensor = current_tensor.to(predicted_tensor.device)

    # 计算变化量的绝对值
    change_magnitude = torch.abs(predicted_tensor - current_tensor)

    # 对于变化量小于阈值的点，保持原值
    adjusted_tensor = torch.where(change_magnitude < threshold,
                                  current_tensor,
                                  predicted_tensor)

    return adjusted_tensor

# ============================================================================
# Visualization Functions
# ============================================================================

def model_performance_visualization(data_list, plot_config, title, step, output_dir):
    """Visualize model predictions."""
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for i, (data, label, config_key) in enumerate(data_list):
        im = axes[i].imshow(data, **plot_config[config_key])
        axes[i].tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    filename = f"{output_dir}/{title.replace(' ', '_')}_step_{step}.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def visualize_temperature_field(data, output_dir, step):
    """Visualize temperature field."""
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    im = axes.imshow(data, cmap='coolwarm',
                     vmin=np.min(data),
                     vmax=np.max(data))

    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.ax.tick_params(labelsize=20)
    axes.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"temp_field_step_{step:06d}.png"), bbox_inches='tight')
    plt.close()

    # 保存色标参考图
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    im = axes.imshow(data, cmap='coolwarm',
                     vmin=np.min(data),
                     vmax=1073)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.ax.tick_params(labelsize=20)
    axes.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"temp_bar.png"), bbox_inches='tight')
    plt.close()


def longrollout_visualization(data_list, plot_config, step, output_dir):
    """Visualize long rollout results."""
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for i, (data, label, config_key) in enumerate(data_list):
        im = axes[i].imshow(data, **plot_config[config_key])
        axes[i].set_title(label, fontsize=14)
        plt.colorbar(im, ax=axes[i])
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rollout_step_{step:06d}.png'), dpi=150)
    plt.close()


# ============================================================================
# Voronoi Generation Functions
# ============================================================================

def poisson_disk_sampling(width, height, min_distance, k=30):
    """Generate Poisson disk samples with a minimum distance."""
    cell_size = min_distance / math.sqrt(2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = {}

    points = []
    active_points = []

    # 添加第一个随机点
    first_point = np.random.rand(2) * [width, height]
    points.append(first_point)
    active_points.append(first_point)

    # 将点添加到网格
    grid_x, grid_y = int(first_point[0] / cell_size), int(first_point[1] / cell_size)
    grid[(grid_x, grid_y)] = [first_point]

    # 持续添加点直到没有活动点
    while active_points:
        idx = np.random.randint(len(active_points))
        point = active_points[idx]

        found = False
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_distance, 2 * min_distance)
            new_point = point + radius * np.array([np.cos(angle), np.sin(angle)])

            # 周期性边界条件处理
            new_point = new_point % [width, height]

            # 检查是否靠近现有点
            grid_x, grid_y = int(new_point[0] / cell_size), int(new_point[1] / cell_size)
            too_close = False

            # 检查周围九宫格
            for i in range(max(0, grid_x - 1), min(grid_width, grid_x + 2)):
                for j in range(max(0, grid_y - 1), min(grid_height, grid_y + 2)):
                    if (i, j) in grid:
                        for p in grid[(i, j)]:
                            dist_vec = np.abs(new_point - p)
                            dist_vec = np.minimum(dist_vec, [width, height] - dist_vec)
                            dist = np.sqrt(np.sum(dist_vec ** 2))
                            if dist < min_distance:
                                too_close = True
                                break
                if too_close:
                    break

            if not too_close:
                points.append(new_point)
                active_points.append(new_point)
                grid.setdefault((grid_x, grid_y), []).append(new_point)
                found = True
                break

        if not found:
            active_points.pop(idx)

    return np.array(points)


def generate_periodic_voronoi(points, domain_size=1.0, num_replicas=1):
    """Generate periodic Voronoi points by tiling."""
    points_list = []

    for i in range(-num_replicas, num_replicas + 1):
        for j in range(-num_replicas, num_replicas + 1):
            shift = np.array([i, j]) * domain_size
            points_list.append(points + shift)

    extended_points = np.vstack(points_list)
    return extended_points


def generate_grain_data(vor, domain_size=1.0, resolution=256):
    """Generate grain microstructure data; GB=0.7, interior=1.0."""
    grid = np.ones((resolution, resolution))

    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution)
    X, Y = np.meshgrid(x, y)
    pixel_positions = np.vstack([X.ravel(), Y.ravel()]).T

    kdtree = KDTree(vor.points)
    _, region_ids = kdtree.query(pixel_positions)

    region_grid = region_ids.reshape(resolution, resolution)

    boundary_mask = np.zeros((resolution, resolution), dtype=bool)

    # 水平晶界
    boundary_mask[:, :-1] |= (region_grid[:, :-1] != region_grid[:, 1:])
    boundary_mask[:, 1:] |= (region_grid[:, :-1] != region_grid[:, 1:])

    # 垂直晶界
    boundary_mask[:-1, :] |= (region_grid[:-1, :] != region_grid[1:, :])
    boundary_mask[1:, :] |= (region_grid[:-1, :] != region_grid[1:, :])

    # 周期性边界
    boundary_mask[:, 0] |= (region_grid[:, 0] != region_grid[:, -1])
    boundary_mask[:, -1] |= (region_grid[:, 0] != region_grid[:, -1])
    boundary_mask[0, :] |= (region_grid[0, :] != region_grid[-1, :])
    boundary_mask[-1, :] |= (region_grid[0, :] != region_grid[-1, :])

    grid[boundary_mask] = 0.62

    return grid


def plot_voronoi(vor, points, domain_size=1.0, point_size=20):
    """Plot Voronoi diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    ax.set_title('周期性沃洛诺伊剖分')

    num_regions = len(vor.point_region)
    colors = np.random.rand(num_regions, 3)

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon = [vor.vertices[v] for v in region]
            ax.fill(*zip(*polygon), color=colors[i % len(colors)], alpha=0.5)

    ax.plot([0, domain_size, domain_size, 0, 0],
            [0, 0, domain_size, domain_size, 0], 'k-', linewidth=2)

    original_points = points % domain_size
    ax.scatter(original_points[:, 0], original_points[:, 1],
               c='black', s=point_size, alpha=0.7)

    plt.tight_layout()
    return fig


def visualize_results(grain_data, smoothed_data, domain_size=1.0):
    """Visualize results with imshow."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    im1 = axes[0, 0].imshow(grain_data, cmap='gray', origin='lower',
                            extent=[0, domain_size, 0, domain_size], vmin=0.6, vmax=1.0)
    axes[0, 0].set_title('原始晶粒数据')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(smoothed_data, cmap='gray', origin='lower',
                            extent=[0, domain_size, 0, domain_size], vmin=0.6, vmax=1.0)
    axes[0, 1].set_title('高斯平滑后 (σ=0.8)')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(grain_data, cmap='viridis', origin='lower',
                            extent=[0, domain_size, 0, domain_size], vmin=0.6, vmax=1.0)
    axes[1, 0].set_title('原始晶粒数据 (viridis)')
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(smoothed_data, cmap='viridis', origin='lower',
                            extent=[0, domain_size, 0, domain_size], vmin=0.6, vmax=1.0)
    axes[1, 1].set_title('高斯平滑后 (viridis)')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    return fig


def check_periodicity(data):
    """Check periodicity by tiling the data."""
    fig, ax = plt.subplots(figsize=(10, 8))

    resolution = data.shape[0]
    tiled = np.block([[data, data], [data, data]])

    im = ax.imshow(tiled, cmap='viridis', origin='lower', vmin=0.6, vmax=1.0)
    ax.axhline(y=resolution, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=resolution, color='r', linestyle='--', alpha=0.5)
    ax.set_title('周期性验证 (2x2 网格)')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def generate_voronoi_data(num_points=40, domain_size=1.0, resolution=256,
                          num_replicas=2, visualize=False,
                          seed=42, output_dir=None):
    """
    Generate Voronoi grain microstructure data.
    """
    np.random.seed(seed)

    # Poisson disk sampling for uniform points
    min_distance = domain_size * 0.8 / math.sqrt(num_points)
    points = poisson_disk_sampling(domain_size, domain_size, min_distance)

    while len(points) < num_points:
        min_distance *= 0.95
        points = poisson_disk_sampling(domain_size, domain_size, min_distance)

    while len(points) > num_points:
        points = points[np.random.choice(len(points), num_points, replace=False)]

    print('Random sampling completed')

    # Create periodic points
    extended_points = generate_periodic_voronoi(points, domain_size, num_replicas)

    # Compute Voronoi tessellation
    vor = Voronoi(extended_points)
    print('Voronoi tessellation completed')

    # Generate grain data
    grain_data = generate_grain_data(vor, domain_size, resolution)
    print('Grain data generation completed')

    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(grain_data, sigma=0.8, mode='wrap')
    print('First smoothing completed')

    # Binarize
    binary_data = np.where(smoothed_data > 0.99, 1, 0.7)
    print('Binarization completed')

    # Smooth again
    smoothed_data = gaussian_filter(binary_data, sigma=0.8, mode='wrap')
    print('Second smoothing completed')

    if visualize:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fig_vor = plot_voronoi(vor, points, domain_size)
        if output_dir:
            fig_vor.savefig(f"{output_dir}/voronoi_diagram.png", dpi=150)
            plt.close(fig_vor)
        else:
            plt.show()
            plt.close()

        fig_result = visualize_results(grain_data, smoothed_data, domain_size)
        if output_dir:
            fig_result.savefig(f"{output_dir}/grain_visualization.png", dpi=150)
            plt.close(fig_result)
        else:
            plt.show()
            plt.close()

        fig_period = check_periodicity(smoothed_data)
        if output_dir:
            fig_period.savefig(f"{output_dir}/periodicity_check.png", dpi=150)
            plt.close(fig_period)
        else:
            plt.show()
            plt.close()

    return smoothed_data


# ============================================================================
# Temperature Field Generation
# ============================================================================

def temperature_field(size, time, heating_rate):
    """
    Generate time-varying temperature field.

    Returns 2D temperature field (923K-1023K)
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Smooth square via L^p norm
    p = 8
    R_square = (np.abs(X) ** p + np.abs(Y) ** p) ** (1 / p)

    # 计算时间因子
    time_factor = min(1.0, time * heating_rate)

    if time_factor >= 1.0:
        # Final state: entire field 1023K
        temperature = np.full((size, size), 1023.0)
    else:
        # Transition state: boundary moves inward
        boundary_position = 1.0 - time_factor
        sharpness = 30

        transition = 1 / (1 + np.exp(sharpness * (R_square - boundary_position)))
        temperature = 1023 - 100 * transition

    return temperature


# ============================================================================
# Main Rollout Function
# ============================================================================

def rollout_with_timevarying_temperature(model, output_dir, grid_size=256, grain_num=40,
                                         max_steps=500, step_interval=50, heating_rate=0.1,
                                         random_seed=42, visualize_init=False):
    """
    Long rollout inference with Voronoi microstructure and time-varying temperature.

    参数:
        model: 深度学习模型
        output_dir: 输出目录
        grid_size: 网格大小
        grain_num: 晶粒数量
        max_steps: 最大推理步数
        step_interval: 可视化间隔
        heating_rate: 温度场加热速率
        random_seed: 随机种子
        visualize_init: 是否可视化初始数据
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "phase_field"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "temperature"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tecplot"), exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Starting rollout inference with time-varying temperature field")
    print(f"Grid size: {grid_size}x{grid_size}, Grain number: {grain_num}")
    print(f"Max steps: {max_steps}, Heating rate: {heating_rate}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    # Generate initial microstructure
    voronoi_dir = os.path.join(output_dir, "init_data") if visualize_init else None
    current_phi = generate_voronoi_data(
        num_points=grain_num,
        resolution=grid_size,
        visualize=visualize_init,
        seed=random_seed,
        output_dir=voronoi_dir
    )

    # Visualization config
    plot_config = {
        "phi": {"vmin": 0.7, "vmax": 1.0, 'cmap': 'viridis'},
        "temp": {"vmin": 923, "vmax": 1023, 'cmap': 'coolwarm'},
        "pdiff": {"vmin": -0.1, "vmax": 0.1, 'cmap': create_bwr_custom_colormap()},
    }

    # Save initial phase field
    data_list = [(current_phi, "Initial Phase Field", "phi")]
    model_performance_visualization(data_list, plot_config, 'Initial_Phase', 0,
                                    os.path.join(output_dir, "phase_field"))
    save_tecplot(current_phi, os.path.join(output_dir, "tecplot", "phase_field_step_000000.dat"),
                 grid_size, grid_size)

    # Physical parameters
    Q = 1.82
    kev = 8.625e-5
    Tmax = 1073.0

    s_time = time.time()

    # Rollout loop
    for t in range(max_steps):
        print(f'Step {t + 1}/{max_steps}', end="\r")

        # Generate current temperature field
        current_temp = temperature_field(grid_size, t / 50, heating_rate)

        # Convert temperature to mobility
        current_temp_m = np.exp(-Q / (kev * current_temp)) / np.exp(-Q / (kev * Tmax))

        # Prepare model input
        input_tensor = torch.tensor(
            np.stack([current_phi, current_temp_m], axis=0),
            dtype=torch.float32
        ).unsqueeze(0).cuda()

        # Model prediction
        with torch.no_grad():
            predicted_phi_scaled, delta_phi = model(input_tensor)
            delta_phi = delta_phi.squeeze().cpu().numpy()

            # Adjust predictions
            predicted_phi = compare_and_adjust(predicted_phi_scaled.squeeze(),
                                               torch.tensor(current_phi.squeeze()).cuda(),
                                               threshold=2e-3)
            predicted_phi = predicted_phi.cpu().numpy()

        # Visualization
        if (t + 1) % step_interval == 0 or t == 0:
            # Phase-field change
            pred_diff = predicted_phi - current_phi

            # Visualize phase field
            data_list = [(predicted_phi, f"Phase Field (Step {t + 1})", "phi")]
            model_performance_visualization(data_list, plot_config, 'Phase_Field', t + 1,
                                            os.path.join(output_dir, "phase_field"))

            # Visualize temperature field
            data_list = [(current_temp, f"Temperature Field (Step {t + 1})", "temp")]
            model_performance_visualization(data_list, plot_config, 'Temperature_Field', t + 1,
                                            os.path.join(output_dir, "temperature"))

            # Save Tecplot
            save_tecplot(predicted_phi, os.path.join(output_dir, "tecplot", f"phase_field_step_{t + 1:06d}.dat"),
                         grid_size, grid_size)

        # Update current phase field
        current_phi = predicted_phi

    e_time = time.time()
    print(f'\nTotal runtime: {e_time - s_time:.4f} s')
    print(f'Avg step time: {(e_time - s_time) / max_steps:.4f} s')
    print(f"Rollout completed. Results saved in: {output_dir}\n")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main: load model and run rollout tests."""

    # =========================================================================
    # CONFIGURATION - 修改这些参数进行实验
    # =========================================================================

    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base_path = os.path.join(repo_root, 'non-isothermal_grain_growth', 'experiment_results', 'augmentation ScaledResNet_C96_NB10_K3_ndt1')
    checkpoint_path = os.path.join(base_path, "best_checkpoint.pt")
    output_base_dir = os.path.join(base_path, "rollout_results_timevarying_temperature")

    # Model parameters (match training)
    base_channels = 96
    num_blocks = 10
    kernel_size = 3

    # Rollout test params
    size_list = [1024]
    max_steps = 1000
    step_interval = 100
    heating_rate = 0.06
    random_seed = 42
    visualize_init = False

    # GPU setup
    gpu_id = 0

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    print(f"\n{'=' * 80}")
    print("Time-Varying Temperature Field Rollout Inference")
    print(f"{'=' * 80}\n")

    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(gpu_id)}")
        torch.cuda.set_device(gpu_id)

    # 创建模型
    print("\nCreating model...")
    model = ResidualConcatNet(
        in_channels=2,
        base_channels=base_channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d,
        min_value=0.7,
        max_value=1.0
    )

    # 加载checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu_id}")
        model.load_state_dict(checkpoint['model_state'])
        print("Successfully loaded model checkpoint")
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_test_loss' in checkpoint:
            print(f"Best test loss: {checkpoint['best_test_loss']:.6f}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 将模型设置为评估模式并移到GPU
    model.cuda()
    model.eval()

    # =========================================================================
    # RUN ROLLOUT TESTS
    # =========================================================================

    print("\nStarting rollout tests with different grid sizes...")

    for size in size_list:
        print(f"\n{'=' * 80}")
        print(f"Testing grid size: {size}x{size}")
        print(f"{'=' * 80}")

        # 为每个size创建输出目录
        output_dir = os.path.join(output_base_dir, f"grid_{size}x{size}")

        try:
            rollout_with_timevarying_temperature(
                model=model,
                output_dir=output_dir,
                grid_size=size,
                grain_num=size * 2,
                max_steps=max_steps,
                step_interval=step_interval,
                heating_rate=heating_rate,
                random_seed=random_seed,
                visualize_init=visualize_init
            )
        except Exception as e:
            print(f"Error during rollout for size {size}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("All rollout tests completed!")
    print(f"Results saved in: {output_base_dir}")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
