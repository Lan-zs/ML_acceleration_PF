"""
Model evaluation utilities for grain growth prediction.

This module provides comprehensive evaluation capabilities including
one-step and rollout prediction analysis with error metrics and visualization.
"""

import os
import time
import joblib
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# from ..utils.visualization import (
#     create_bwr_custom_colormap,
#     create_wr_custom_colormap,
#     model_performance_visualization,
#     error_brokenaxes_plot,
#     error_plot,
#     model_performance_visualization11
# )  # 路径错误引用不了

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.visualization import (
    create_bwr_custom_colormap,
    create_wr_custom_colormap,
    model_performance_visualization,
    error_brokenaxes_plot,
    error_plot,
    model_performance_visualization11,
    model_performance_visualization13
)


def save_tecplot(phi_slice: np.ndarray, output_path: str, I: int, J: int) -> None:
    """
    Save phase field data in Tecplot format.

    Args:
        phi_slice: 2D array of phase field data for single time step.
        output_path: Output file path.
        I, J: Grid dimensions.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        # Write data
        for value in phi_slice.flatten():
            f.write(f"{value:.6f}\n")


def compare_and_adjust(predicted_tensor, current_tensor, threshold=1e-2):
    """
    比较预测值与当前值之间的变化量，对于变化量小于阈值的点保持原值

    参数:
        predicted_tensor (torch.Tensor): 模型预测的张量
        current_tensor (torch.Tensor): 当前的张量值
        threshold (float): 变化量阈值，默认为1e-2

    返回:
        torch.Tensor: 校正后的张量
    """
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


def model_testing(
        model: nn.Module,
        phi_data: np.ndarray,
        temp_data: np.ndarray,
        indices: List[int],
        output_dir: str,
        evaluation_mode: str,
        assess_metrics: str = 'MAE',
        step_interval: int = 50,
        ndt: int = 1
) -> Dict:
    """
    Comprehensive model testing with error analysis and visualization.

    Args:
        model: Trained neural network model.
        phi_data: Phase field evolution data.
        temp_data: Temperature field data.
        indices: Time indices for detailed analysis.
        output_dir: Output directory for results.
        evaluation_mode: 'onestep' or 'rollout' evaluation mode.
        assess_metrics: Error metric ('MAE' or 'MSE').
        step_interval: Interval for visualization.
        ndt: Time step interval.

    Returns:
        dict: Dictionary containing error statistics.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tracking variables
    time_steps, I, J = phi_data.shape
    current_phi = phi_data[0]  # Initial phase field

    # Error tracking lists
    total_errors = []
    unchanged_errors = []
    changed_errors = []
    zero_errors = []
    changed_zero_errors = []
    unchanged_ratios = []
    changed_ratios = []

    start_time = time.time()

    print(f"Starting {evaluation_mode} evaluation with {assess_metrics} metrics")

    # Iterate through time steps
    for t in range(0, time_steps - ndt, ndt):
        print(f'Progress: {t / time_steps:.1%}', end="\r")

        # Get ground truth data
        current_true_phi = phi_data[t].squeeze()
        next_true_phi = phi_data[t + ndt].squeeze()

        # # Identify unchanged and changed regions
        unchanged_mask = (np.abs(current_true_phi - next_true_phi) < 1e-6)
        changed_mask = ~unchanged_mask
        # unchanged_mask = (np.abs(current_true_phi - next_true_phi) < 1e-3)
        # changed_mask = ~unchanged_mask

        # Calculate region ratios
        unchanged_ratio = np.mean(unchanged_mask)
        changed_ratio = 1 - unchanged_ratio
        unchanged_ratios.append(unchanged_ratio)
        changed_ratios.append(changed_ratio)

        # Prepare model input
        input_tensor = torch.tensor(
            np.stack([current_phi.squeeze(), temp_data], axis=0), dtype=torch.float32
        ).unsqueeze(0).cuda()

        # Model prediction
        with torch.no_grad():
            predicted_phi_scaled, delta_phi = model(input_tensor)
            predicted_phi = predicted_phi_scaled.squeeze().cpu().numpy()
            delta_phi = delta_phi.squeeze().cpu().numpy()

            if evaluation_mode == 'rollout':
                # # 新的校正方法：比较变化量，对于变化量小于阈值的点保持原值
                predicted_phi = compare_and_adjust(predicted_phi_scaled.squeeze(),
                                                   torch.tensor(current_phi.squeeze()).cuda(),
                                                   threshold=2e-3)
                predicted_phi = predicted_phi.cpu().numpy()


        # Calculate errors based on specified metric
        if assess_metrics == 'MAE':
            # Zero baseline error (predicting no change)
            zero_error = np.mean(np.abs(next_true_phi - current_true_phi))
            zero_errors.append(zero_error)

            # Total prediction error
            total_error = np.mean(np.abs(predicted_phi - next_true_phi))
            total_errors.append(total_error)

            # Unchanged region error
            if np.sum(unchanged_mask) > 0:
                unchanged_error = np.sum(
                    np.abs(predicted_phi[unchanged_mask] - next_true_phi[unchanged_mask])
                ) / np.sum(unchanged_mask)
            else:
                unchanged_error = 0
            unchanged_errors.append(unchanged_error)

            # Changed region error
            if np.sum(changed_mask) > 0:
                changed_error = np.sum(
                    np.abs(predicted_phi[changed_mask] - next_true_phi[changed_mask])
                ) / np.sum(changed_mask)
                changed_zero_error = np.sum(
                    np.abs(current_true_phi[changed_mask] - next_true_phi[changed_mask])
                ) / np.sum(changed_mask)
            else:
                changed_error = 0
                changed_zero_error = 0
            changed_errors.append(changed_error)
            changed_zero_errors.append(changed_zero_error)

        elif assess_metrics == 'MSE':
            # MSE-based error calculations
            zero_error = np.mean((next_true_phi - current_true_phi) ** 2)
            zero_errors.append(zero_error)

            total_error = np.mean((predicted_phi - next_true_phi) ** 2)
            total_errors.append(total_error)

            if np.sum(unchanged_mask) > 0:
                unchanged_error = np.sum(
                    (predicted_phi[unchanged_mask] - next_true_phi[unchanged_mask]) ** 2
                ) / np.sum(unchanged_mask)
            else:
                unchanged_error = 0
            unchanged_errors.append(unchanged_error)

            if np.sum(changed_mask) > 0:
                changed_error = np.sum(
                    (predicted_phi[changed_mask] - next_true_phi[changed_mask]) ** 2
                ) / np.sum(changed_mask)
                changed_zero_error = np.sum(
                    (current_true_phi[changed_mask] - next_true_phi[changed_mask]) ** 2
                ) / np.sum(changed_mask)
            else:
                changed_error = 0
                changed_zero_error = 0
            changed_errors.append(changed_error)
            changed_zero_errors.append(changed_zero_error)

        # Save specific time steps as Tecplot files
        if t in indices:
            data_list = [
                (predicted_phi, f"Predicted_Phi{t + ndt}.dat"),
                (next_true_phi, f"True_Phi{t + ndt}.dat"),
            ]

            for data, filename in data_list:
                file_path = os.path.join(output_dir, filename)
                save_tecplot(data, file_path, I, J)

        # Visualization at specified intervals
        if t % step_interval == 0:
            # Calculate difference metrics
            abs_diff = np.abs(predicted_phi - next_true_phi)
            diff = predicted_phi - next_true_phi
            pred_change = predicted_phi - current_phi.squeeze()
            true_change = next_true_phi - current_true_phi

            # Calculate loss for display
            criterion = nn.L1Loss()
            loss = criterion(
                torch.from_numpy(predicted_phi),
                torch.from_numpy(next_true_phi)
            )

            # Setup visualization configuration
            plot_config = {
                "p_phi": {
                    "vmin": np.min(predicted_phi),
                    "vmax": np.max(predicted_phi),
                    'cmap': 'viridis'
                },
                "t_phi": {
                    "vmin": np.min(next_true_phi),
                    "vmax": np.max(next_true_phi),
                    'cmap': 'viridis'
                },
                "adiff": {
                    "vmin": 0,
                    "vmax": np.max(abs_diff),
                    'cmap': create_wr_custom_colormap()
                },
                "diff": {
                    "vmin": -np.max(abs_diff),
                    "vmax": np.max(abs_diff),
                    'cmap': create_bwr_custom_colormap()
                },
                "pdiff": {
                    "vmin": -np.max(np.abs(pred_change)),
                    "vmax": np.max(np.abs(pred_change)),
                    'cmap': create_bwr_custom_colormap()
                },
                "tdiff": {
                    "vmin": -np.max(np.abs(true_change)),
                    "vmax": np.max(np.abs(true_change)),
                    'cmap': create_bwr_custom_colormap()
                },
            }

            # Data for visualization
            data_list = [
                (predicted_phi, "Predicted Phi", "p_phi"),
                (next_true_phi, "True Phi", "t_phi"),
                (abs_diff, f"Absolute Difference (MAE: {loss.item():.4e})", "adiff"),
                (pred_change, "Predicted Change", "pdiff"),
                (true_change, "True Change", "tdiff"),
                (diff, "Difference (Predicted - True)", "diff"),
            ]

            model_performance_visualization(data_list, plot_config, evaluation_mode, t, output_dir)

        # Update current state based on evaluation mode
        if evaluation_mode == 'rollout':
            current_phi = predicted_phi
        elif evaluation_mode == 'onestep':
            current_phi = next_true_phi

    end_time = time.time()
    print(f'\nEvaluation completed in {end_time - start_time:.2f} seconds')

    # Compile error data
    error_data = {
        'total_errors': total_errors,
        'unchanged_errors': unchanged_errors,
        'changed_errors': changed_errors,
        'zero_errors': zero_errors,
        'changed_zero_errors': changed_zero_errors,
        'unchanged_ratios': unchanged_ratios,
        'changed_ratios': changed_ratios,
    }

    # Save error data
    error_file = os.path.join(output_dir, "errors.joblib")
    joblib.dump(error_data, error_file)

    # Generate error plots
    if evaluation_mode == 'onestep':
        error_brokenaxes_plot(
            total_errors, unchanged_errors, changed_errors,
            output_dir, evaluation_mode, assess_metrics,
            zero_errors=zero_errors, changed_zero_errors=changed_zero_errors
        )
        error_plot(
            total_errors, unchanged_errors, changed_errors,
            output_dir, evaluation_mode, assess_metrics, output_dir,
            zero_errors=zero_errors, changed_zero_errors=changed_zero_errors
        )
    elif evaluation_mode == 'rollout':
        error_brokenaxes_plot(
            total_errors, unchanged_errors, changed_errors,
            output_dir, evaluation_mode, assess_metrics
        )
        error_plot(
            total_errors, unchanged_errors, changed_errors,
            output_dir, evaluation_mode, assess_metrics, output_dir
        )

    return error_data



if __name__ == '__main__':
    pass
