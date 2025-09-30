import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
import torch.nn as nn
import random
import torch
import numpy as np
import h5py
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet,ResidualConcatNet_Res

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_wr_custom_colormap():
    """Create a white-to-red custom colormap."""
    return LinearSegmentedColormap.from_list("wr_cmap", ["white", "red"])



def save_tecplot(phi_slice, output_path, I, J):
    """Save a single time-step phi slice in Tecplot format."""
    with open(output_path, 'w') as f:
        # 写入头部信息
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        # 写入数据
        for value in phi_slice.flatten():
            f.write(f"{value:.6f}\n")


def calculate_phase_fractions(phi_data):
    """Compute volume fractions of two phases."""
    phase1_mask = (phi_data <= 0.6)
    phase2_mask = (phi_data > 0.6)
    phase1_fraction = np.mean(phase1_mask)
    phase2_fraction = np.mean(phase2_mask)
    return phase1_fraction, phase2_fraction


def visualize_relative_error(relative_errors, time_steps, model_transitions, output_dir):
    """Visualize relative error over time."""
    plt.figure(figsize=(12, 6))

    # 绘制相对误差
    plt.plot(time_steps, relative_errors, 'b-', linewidth=2)

    # 标记模型转换点
    for transition, label in model_transitions:
        plt.axvline(x=transition, color='r', linestyle='--', alpha=0.7)
        plt.text(transition, max(relative_errors) * 0.9, label, rotation=90, alpha=0.7)

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Mean Relative Error', fontsize=12)
    plt.title('Relative Error Evolution during Multi-scale Model Rollout', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_error_evolution.png"), dpi=150)
    plt.close()

def visualize_phase_fractions(phase1_true, phase2_true, phase1_pred, phase2_pred, time_steps, output_dir):
    """Visualize phase volume fractions over time."""
    import matplotlib.pyplot as plt

    # 创建图像和双y轴
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # 获取viridis颜色映射的颜色
    cmap = plt.cm.viridis
    color1 = cmap(0.01)  # 第一相颜色 (接近最小值)
    color2 = cmap(0.99)  # 第二相颜色 (接近最大值)

    # 绘制相1的体积分数
    ax1.plot(time_steps,phase1_true, '--', color=color1, linewidth=2, label='Phase 1 (True)')
    ax1.plot(time_steps,phase1_pred, '-', color=color1, linewidth=2, label='Phase 1 (Predicted)')

    # 绘制相2的体积分数
    ax2.plot(time_steps,phase2_true, '--', color=color2, linewidth=2, label='Phase 2 (True)')
    ax2.plot(time_steps,phase2_pred, '-', color=color2, linewidth=2, label='Phase 2 (Predicted)')

    # 设置轴标签和标题
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Phase 1 Volume Fraction', fontsize=12)
    ax2.set_ylabel('Phase 2 Volume Fraction', fontsize=12)
    plt.title(f'Phase Volume Fractions Evolution', fontsize=14)

    # 设置左右轴的刻度范围为0到1
    # ax1.set_ylim(0.4, 0.6)
    # ax2.set_ylim(0.4, 0.6)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # 设置图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    # 设置网格和刻度颜色
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'phase_fractions.png'), dpi=150, bbox_inches='tight')
    plt.close()

def model_performance_visualization_custom(pred_data, true_data, diff_data, timestep, output_dir):
    """Custom visualization for predicted, true, and absolute error."""
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 14))

    # 计算全局最小值和最大值，确保相同的颜色范围
    vmin = min(np.min(pred_data), np.min(true_data))
    vmax = max(np.max(pred_data), np.max(true_data))

    # 预测数据
    im1 = axes[0].imshow(pred_data, cmap='viridis', vmin=vmin, vmax=vmax)
    # axes[0].set_title(f'Predicted Phi (t={timestep})', fontsize=14)
    cbar = plt.colorbar(im1, ax=axes[0], shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    axes[0].tick_params(axis='both', which='major', labelsize=20)

    # 真实数据
    im2 = axes[1].imshow(true_data, cmap='viridis', vmin=vmin, vmax=vmax)
    # axes[1].set_title(f'True Phi (t={timestep})', fontsize=14)
    cbar = plt.colorbar(im2, ax=axes[1], shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)

    # 绝对误差
    im3 = axes[2].imshow(diff_data, cmap=create_wr_custom_colormap(), vmin=0, vmax=np.max(diff_data))
    # axes[2].set_title(f'Absolute Difference (t={timestep})', fontsize=14)
    cbar = plt.colorbar(im3, ax=axes[2], shrink=0.9)
    cbar.ax.tick_params(labelsize=20)
    axes[2].tick_params(axis='both', which='major', labelsize=20)



    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_t{timestep}.png"), dpi=150)
    plt.close()


def multi_stage_model_prediction(model_10dt, model_100dt, model_500dt,
                                 h5_file_path1, h5_file_path2, h5_file_path3,
                                 output_dir):
    """Run three-scale model rollout and compare with ground truth."""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)


    # 读取三个H5文件
    print("Reading H5 files ...")

    # 读取第一个文件 (1dt, 0-2000步)
    with h5py.File(h5_file_path1, 'r') as f:
        phi_data1 = f['phi_data'][:]
        print(f"File 1 data shape: {phi_data1.shape}")

    # 读取第二个文件 (100dt, 2100-50000步)
    with h5py.File(h5_file_path2, 'r') as f:
        phi_data2 = f['phi_data'][:]
        print(f"File 2 data shape: {phi_data2.shape}")

    # 读取第三个文件 (500dt, 50500-200000步)
    with h5py.File(h5_file_path3, 'r') as f:
        phi_data3 = f['phi_data'][:]
        print(f"File 3 data shape: {phi_data3.shape}")

    # 获取初始噪声场
    initial_phi = phi_data1[0]
    I, J = initial_phi.shape

    # 保存初始场
    save_tecplot(initial_phi, os.path.join(output_dir, "initial_phi.dat"), I, J)

    # Model switch points
    MODEL_SWITCH_1 = 2000
    MODEL_SWITCH_2 = 50000
    # MODEL_SWITCH_2 = 2100  # 从100dt切换到500dt的时间步
    # MODEL_SWITCH_2 = 5000  # 从100dt切换到500dt的时间步

    # 初始化数组用于存储结果
    relative_errors = []  # 相对误差
    phase1_true = []  # 真实相1体积分数
    phase2_true = []  # 真实相2体积分数
    phase1_pred = []  # 预测相1体积分数
    phase2_pred = []  # 预测相2体积分数
    time_steps = []  # 时间步
    time_steps_phase = []  # 时间步

    # Current phi
    current_phi = initial_phi.copy()

    # Special timesteps to visualize
    special_timesteps = [0, 20, 600, 2000, 30000, 40000, 50000, 100000, 200000]

    # 开始计时
    start_time = time.time()

    print("\nStage 1: 10dt model (0-2000 steps)")
    # Stage 1
    current_step = 0
    while current_step < MODEL_SWITCH_1:
        # One model prediction per 10 steps
        target_step = current_step + 10

        if current_step==0:
            model_performance_visualization_custom(
                current_phi, current_phi, current_phi,
                0, output_dir
            )

        # Prepare model input
        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        # Predict with 10dt model
        with torch.no_grad():
            predicted_phi_scaled, _ = model_10dt(input_tensor)
            predicted_phi = predicted_phi_scaled.squeeze().cpu().numpy()

        # Get ground truth
        true_phi_index = target_step
        if true_phi_index < len(phi_data1):
            true_phi = phi_data1[true_phi_index]

            # Compute relative error
            epsilon = 1e-8
            abs_error = np.abs(predicted_phi - true_phi)
            mean_relative_error = np.sum(abs_error) / (np.sum(np.abs(true_phi)) + epsilon)

            # Store results
            relative_errors.append(mean_relative_error)
            time_steps.append(target_step)

            # Volume fraction during spinodal separation not used in Stage 1

            # p1_true, p2_true = calculate_phase_fractions(true_phi)
            # p1_pred, p2_pred = calculate_phase_fractions(predicted_phi)

            # phase1_true.append(p1_true)
            # phase2_true.append(p2_true)
            # phase1_pred.append(p1_pred)
            # phase2_pred.append(p2_pred)

            # Visualize at special timesteps
            if target_step in special_timesteps:
                print(f"处理特殊时间步: {target_step}")
                model_performance_visualization_custom(
                    predicted_phi, true_phi, abs_error,
                    target_step, output_dir
                )

        # Update current state
        current_phi = predicted_phi
        current_step = target_step

        # Progress
        if current_step % 100 == 0:
            print(f"Stage 1: {current_step}/{MODEL_SWITCH_1}", end="\r")

    # 保存最终预测和真实相场
    save_tecplot(predicted_phi, os.path.join(output_dir, "2k_predicted_phi.dat"), I, J)
    save_tecplot(true_phi, os.path.join(output_dir, "2k_true_phi.dat"), I, J)

    print("\n\nStage 2: 100dt model (2000-50000 steps)")
    # Stage 2
    while current_step < MODEL_SWITCH_2:
        # One prediction per 100 steps
        target_step = current_step + 100


        # Prepare input
        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        # Predict with 100dt model
        with torch.no_grad():
            predicted_phi_scaled, _ = model_100dt(input_tensor)
            predicted_phi = predicted_phi_scaled.squeeze().cpu().numpy()

        # Get ground truth
        if target_step >= 2100:  # 第二个文件从2100步开始
            true_phi_index = (target_step - 2100) // 100
            if true_phi_index < len(phi_data2):
                true_phi = phi_data2[true_phi_index]

                # 计算相对误差
                epsilon = 1e-8
                abs_error = np.abs(predicted_phi - true_phi)
                mean_relative_error = np.sum(abs_error) / (np.sum(np.abs(true_phi)) + epsilon)

                # 存储结果
                relative_errors.append(mean_relative_error)
                time_steps.append(target_step)
                time_steps_phase.append((target_step))

                # Compute phase volume fractions
                p1_true, p2_true = calculate_phase_fractions(true_phi)
                p1_pred, p2_pred = calculate_phase_fractions(predicted_phi)

                phase1_true.append(p1_true)
                phase2_true.append(p2_true)
                phase1_pred.append(p1_pred)
                phase2_pred.append(p2_pred)

                # Visualize at special timesteps
                if target_step in special_timesteps:
                    print(f"处理特殊时间步: {target_step}")
                    model_performance_visualization_custom(
                        predicted_phi, true_phi, abs_error,
                        target_step, output_dir
                    )


        # Update state
        current_phi = predicted_phi
        current_step = target_step

        # Progress
        if current_step % 1000 == 0:
            print(f"Stage 2: {current_step}/{MODEL_SWITCH_2}", end="\r")

    # 保存最终预测和真实相场
    save_tecplot(predicted_phi, os.path.join(output_dir, "5w_predicted_phi.dat"), I, J)
    save_tecplot(true_phi, os.path.join(output_dir, "5w_true_phi.dat"), I, J)

    print("\n\nStage 3: 500dt model (50000-200000 steps)")
    # Stage 3
    MAX_STEP = 200000
    while current_step < MAX_STEP:
        # One prediction per 500 steps
        target_step = current_step + 500


        # Prepare input
        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        # Predict with 500dt model
        with torch.no_grad():
            predicted_phi_scaled, _ = model_500dt(input_tensor)
            predicted_phi = predicted_phi_scaled.squeeze().cpu().numpy()

        # Get truth
        if target_step >= 50500:  # 第三个文件从50500步开始
            true_phi_index = (target_step - 50500) // 500
            if true_phi_index < len(phi_data3):
                true_phi = phi_data3[true_phi_index]

                # 计算相对误差
                epsilon = 1e-8
                abs_error = np.abs(predicted_phi - true_phi)
                mean_relative_error = np.sum(abs_error) / (np.sum(np.abs(true_phi)) + epsilon)

                # 存储结果
                relative_errors.append(mean_relative_error)
                time_steps.append(target_step)
                time_steps_phase.append(target_step)

                # Compute phase volume fractions
                p1_true, p2_true = calculate_phase_fractions(true_phi)
                p1_pred, p2_pred = calculate_phase_fractions(predicted_phi)

                phase1_true.append(p1_true)
                phase2_true.append(p2_true)
                phase1_pred.append(p1_pred)
                phase2_pred.append(p2_pred)

                # Visualize at special timesteps
                if target_step in special_timesteps:
                    print(f"处理特殊时间步: {target_step}")
                    model_performance_visualization_custom(
                        predicted_phi, true_phi, abs_error,
                        target_step, output_dir
                    )


        # Update state
        current_phi = predicted_phi
        current_step = target_step

        # Progress
        if current_step % 10000 == 0:
            print(f"Stage 3: {current_step}/{MAX_STEP}", end="\r")

    end_time = time.time()
    print(f"\n\nAll inference completed. Total time: {end_time - start_time:.2f} s")

    # 获取最终的真实数据
    final_true_phi = phi_data3[-1]  # 最后一帧

    # 保存最终预测和真实相场
    save_tecplot(predicted_phi, os.path.join(output_dir, "final_predicted_phi.dat"), I, J)
    save_tecplot(final_true_phi, os.path.join(output_dir, "final_true_phi.dat"), I, J)

    # Define model transition points for visualization
    model_transitions = [
        (MODEL_SWITCH_1, "10dt→100dt"),
        (MODEL_SWITCH_2, "100dt→500dt")
    ]

    # 可视化相对误差
    visualize_relative_error(relative_errors, time_steps, model_transitions, output_dir)

    # 可视化相体积分数
    visualize_phase_fractions(phase1_true, phase2_true, phase1_pred, phase2_pred,
                                     time_steps_phase, output_dir)

    # 保存结果数据
    results = {
        'relative_errors': relative_errors,
        'time_steps': time_steps,
        'time_steps_phase': time_steps_phase,
        'phase1_true': phase1_true,
        'phase2_true': phase2_true,
        'phase1_pred': phase1_pred,
        'phase2_pred': phase2_pred
    }

    joblib.dump(results, os.path.join(output_dir, "multi_stage_results.joblib"))

    return results



if __name__ == '__main__':
    save_state = True
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(repo_root, 'spinodal_decomposition', 'spi_noise_rollout_3stage')


    # 设置随机数种子
    setup_seed(1234)
    # setup_seed(666)

    # 添加这行来支持多进程
    torch.multiprocessing.freeze_support()

    # 打印可用的GPU设备
    print("CUDA Available: ", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

    model_500dt = ResidualConcatNet(
        in_channels=1,
        base_channels=32,
        num_blocks=10,
        kernel_size=3,
        act_fn=nn.GELU,  # GELU activation function
        norm_2d=nn.BatchNorm2d,
        min_value=0.175,
        max_value=0.945
    )

    model_100dt = ResidualConcatNet(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=nn.GELU,  # GELU activation function
        norm_2d=nn.BatchNorm2d,
        min_value=0.175,
        max_value=0.945
    )

    model_10dt = ResidualConcatNet_Res(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=nn.GELU,  # GELU activation function
        norm_2d=nn.BatchNorm2d,
    )

    torch.cuda.set_device(1)
    model_10dt.cuda()  # 可能有问题，后面试一试
    model_100dt.cuda()  # 可能有问题，后面试一试
    model_500dt.cuda()  # 可能有问题，后面试一试

    # Load best model checkpoints
    checkpoint_10dt = torch.load(os.path.join(repo_root, 'spinodal_decomposition', '10dt_model_results', 'augmentation ResResNet_C32_NB8_K3_ndt10', 'best_checkpoint.pt'), map_location='cuda:1')
    checkpoint_100dt = torch.load(os.path.join(repo_root, 'spinodal_decomposition', '100dt_model_results_min175', 'augmentation ScaledResNet_C32_NB8_K3_ndt1', 'best_checkpoint.pt'), map_location='cuda:1')
    checkpoint_500dt = torch.load(os.path.join(repo_root, 'spinodal_decomposition', '500dt_model_results', 'augmentation ScaledResNet_C32_NB10_K3_ndt5', 'best_checkpoint.pt'), map_location='cuda:1')

    model_10dt.load_state_dict(checkpoint_10dt['model_state'])
    model_100dt.load_state_dict(checkpoint_100dt['model_state'])
    model_500dt.load_state_dict(checkpoint_500dt['model_state'])

    # 测试阶段
    model_10dt.eval()
    model_100dt.eval()
    model_500dt.eval()


    # H5 paths
    h5_file_path1 = os.path.join(repo_root, 'dataset', 'spinodal_decomposition_3stage', 'S375-1_test1024-1dt_I1024_J1024_ndt1_mint0_maxt2000.h5')
    h5_file_path2 = os.path.join(repo_root, 'dataset', 'spinodal_decomposition_3stage', 'S375-1_test1024_I1024_J1024_ndt1_mint21_maxt500.h5')
    h5_file_path3 = os.path.join(repo_root, 'dataset', 'spinodal_decomposition_3stage', 'S375-1_test1024_I1024_J1024_ndt5_mint505_maxt2000.h5')


    # 执行三阶段模型预测
    results = multi_stage_model_prediction(
        model_10dt, model_100dt, model_500dt,
        h5_file_path1, h5_file_path2, h5_file_path3,
        output_dir
    )