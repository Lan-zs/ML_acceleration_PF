"""
Visualization utilities for grain growth prediction models.

This module provides comprehensive visualization tools for model performance,
training progress, and error analysis.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Any, Optional

def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_wr_custom_colormap():
    """
    Create a custom white-to-red colormap.

    Returns:
        LinearSegmentedColormap: Custom colormap from white to red.
    """
    colors = ["white", "red"]
    return LinearSegmentedColormap.from_list("white_red", colors, N=256)


def create_bwr_custom_colormap():
    """
    Create a custom blue-white-red colormap.

    Returns:
        LinearSegmentedColormap: Custom colormap with blue-white-red gradient.
    """
    colors = ["blue", "white", "red"]
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


def model_performance_during_training(
        current_phi: torch.Tensor,
        delta_phi: torch.Tensor,
        predicted_phi: torch.Tensor,
        next_phi: torch.Tensor,
        epoch: int,
        loss: float,
        mode: str,
        save_dir: str
) -> None:
    """
    Visualize model performance during training.

    Args:
        current_phi: Current phase field.
        delta_phi: Raw model output (delta phi).
        predicted_phi: Scaled model prediction.
        next_phi: Ground truth next phase field.
        epoch: Current training epoch.
        loss: Current loss value.
        mode: Training mode identifier.
        save_dir: Directory to save visualizations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Extract numpy arrays from tensors
    dp = delta_phi[0].squeeze().cpu().detach().numpy()
    dp_scaled = predicted_phi[0].squeeze().cpu().detach().numpy() - current_phi[0].squeeze().cpu().detach().numpy()
    true_change = next_phi[0].squeeze().cpu().detach().numpy() - current_phi[0].squeeze().cpu().detach().numpy()

    # Plot raw delta phi
    im1 = axes[0].imshow(
        dp,
        cmap=create_bwr_custom_colormap(),
        vmin=-np.max(np.abs(dp)),
        vmax=np.max(np.abs(dp))
    )
    plt.colorbar(im1, ax=axes[0], shrink=0.7)
    axes[0].set_title(f"{mode}: Raw Delta Phi")

    # Plot scaled delta phi
    max_change = np.max(np.abs(true_change))
    im2 = axes[1].imshow(
        dp_scaled,
        cmap=create_bwr_custom_colormap(),
        vmin=-max_change,
        vmax=max_change
    )
    plt.colorbar(im2, ax=axes[1], shrink=0.7)
    axes[1].set_title(f"{mode}: Scaled Delta Phi")

    # Plot true change
    im3 = axes[2].imshow(
        true_change,
        cmap=create_bwr_custom_colormap(),
        vmin=-max_change,
        vmax=max_change
    )
    plt.colorbar(im3, ax=axes[2], shrink=0.7)
    axes[2].set_title(f"{mode}: True Delta Phi")

    plt.suptitle(f"{mode}: Epoch {epoch + 1} | Loss: {loss:.3e}", fontsize=15)
    plt.tight_layout()

    # Save figure
    filename = os.path.join(save_dir, f"{mode}_prediction_epoch{epoch}.png")
    plt.savefig(filename)
    plt.close()


def plot_train_loss_curve(
        train_losses: List[float],
        test_losses: List[float],
        best_losses: List[float],
        optimizer_lrs: List[float],
        save_dir: str,
        experiment_name: str
) -> None:
    """
    Plot training loss curves.

    Args:
        train_losses: Training loss values.
        test_losses: Validation loss values.
        best_losses: Best loss values.
        optimizer_lrs: Learning rate values.
        save_dir: Directory to save the plot.
        experiment_name: Name of the experiment.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.plot(best_losses, label="Best Loss")
    plt.plot(optimizer_lrs, label="Learning Rate")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f'Loss Curve: {experiment_name} | Best Loss: {np.min(best_losses):.3e}')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


def model_performance_visualization(
        data_list: List[Tuple[np.ndarray, str, str]],
        plot_config: Dict[str, Dict[str, Any]],
        evaluation_mode: str,
        step: int,
        output_dir: str
) -> None:
    """
    Visualize model performance with multiple subplots.

    Args:
        data_list: List of (data, title, config_key) tuples.
        plot_config: Configuration for visualization parameters.
        evaluation_mode: Evaluation mode identifier.
        step: Current time step.
        output_dir: Output directory for saving.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for ax, (data, title, config_key) in zip(axes.flat, data_list):
        config = plot_config[config_key]
        im = ax.imshow(
            data,
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        ax.set_title(title, fontsize=20)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"{evaluation_mode.title()} - Step {step:06d}", fontsize=25)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"step_{step:06d}.png")
    plt.savefig(filename)
    plt.close()


def model_performance_visualization11(
        data_list: List[Tuple[np.ndarray, str, str]],
        plot_config: Dict[str, Dict[str, Any]],
        evaluation_mode: str,
        step: int,
        output_dir: str
) -> None:
    """
    Single subplot visualization for model performance.

    Args:
        data_list: List of (data, title, config_key) tuples.
        plot_config: Configuration for visualization parameters.
        evaluation_mode: Evaluation mode identifier.
        step: Current time step.
        output_dir: Output directory for saving.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for (data, title, config_key) in data_list:
        config = plot_config[config_key]
        im = ax.imshow(
            data,
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        ax.set_title(title, fontsize=20)

    plt.suptitle(f"{evaluation_mode} - Step {step:06d}", fontsize=25)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"step11_{step:06d}.png")
    plt.savefig(filename)
    plt.close()

def model_performance_visualization13(
        data_list: List[Tuple[np.ndarray, str, str]],
        plot_config: Dict[str, Dict[str, any]],
        evaluation_mode: str,
        step: int,
        output_dir: str
) -> None:
    """
    Visualization for multi-frame rollout analysis with 3 subplots layout.

    Args:
        data_list: List of (data, title, config_key) tuples.
        plot_config: Configuration for visualization parameters.
        evaluation_mode: Evaluation mode identifier.
        step: Current time step.
        output_dir: Output directory for saving.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (data, title, config_key) in zip(axes.flat, data_list):
        config = plot_config[config_key]
        im = ax.imshow(
            data,
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        ax.set_title(title, fontsize=16)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"{evaluation_mode.title()} - Step {step:06d}", fontsize=20)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"step_{step:06d}.png")
    plt.savefig(filename, dpi=150)
    plt.close()

def error_plot(
        total_errors: List[float],
        unchanged_errors: List[float],
        changed_errors: List[float],
        output_dir: str,
        evaluation_mode: str,
        assess_metrics: str,
        experiment_name: str,
        zero_errors: Optional[List[float]] = None,
        changed_zero_errors: Optional[List[float]] = None
) -> None:
    """
    Plot error analysis results.

    Args:
        total_errors: Total prediction errors.
        unchanged_errors: Errors in unchanged regions.
        changed_errors: Errors in changed regions.
        output_dir: Output directory for saving.
        evaluation_mode: Evaluation mode.
        assess_metrics: Assessment metric used.
        experiment_name: Name of experiment.
        zero_errors: Zero baseline errors (optional).
        changed_zero_errors: Zero baseline errors for changed regions (optional).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Plot baseline errors if provided
    if zero_errors is not None:
        ax.plot(zero_errors, label="Zero Baseline", color='blue', linestyle='--')
    if changed_zero_errors is not None:
        ax.plot(changed_zero_errors, label="Changed Zero Baseline", color='red', linestyle='--')

    # Plot prediction errors
    ax.plot(total_errors, label="Total Error", color='blue')
    ax.plot(unchanged_errors, label="Unchanged Region Error", color='green')
    ax.plot(changed_errors, label="Changed Region Error", color='red')

    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"Mean {assess_metrics} Error")
    ax.set_title(f"({assess_metrics} error: {np.mean(total_errors):.2e})")
    ax.set_yscale('log')

    plt.suptitle(f"{experiment_name} {evaluation_mode.title()}", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "error_analysis.png"))
    plt.close()


def error_brokenaxes_plot(
        total_errors: List[float],
        unchanged_errors: List[float],
        changed_errors: List[float],
        output_dir: str,
        evaluation_mode: str,
        experiment_name: str,
        zero_errors: Optional[List[float]] = None,
        changed_zero_errors: Optional[List[float]] = None
) -> None:
    """
    Create broken axis error plot for better visualization of different error scales.

    Args:
        total_errors: Total prediction errors.
        unchanged_errors: Errors in unchanged regions.
        changed_errors: Errors in changed regions.
        output_dir: Output directory for saving.
        evaluation_mode: Evaluation mode.
        experiment_name: Name of experiment.
        zero_errors: Zero baseline errors (optional).
        changed_zero_errors: Zero baseline errors for changed regions (optional).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[2, 0.5])

    # Plot upper part
    if zero_errors is not None:
        ax1.plot(zero_errors, label="Zero Baseline", color='blue', linestyle='--')
    if changed_zero_errors is not None:
        ax1.plot(changed_zero_errors, label="Changed Zero Baseline", color='red', linestyle='--')

    ax1.plot(total_errors, label="Total Error", color='blue')
    ax1.plot(unchanged_errors, label="Unchanged Region Error", color='green')
    ax1.plot(changed_errors, label="Changed Region Error", color='red')
    ax1.set_ylim(2e-5, 1.3e-2)
    ax1.set_yscale('log')

    # Hide bottom spines and x-axis
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_visible(False)

    # Plot lower part
    ax2.plot(total_errors, label="Total Error", color='blue')
    ax2.plot(unchanged_errors, label="Unchanged Region Error", color='green')
    ax2.plot(changed_errors, label="Changed Region Error", color='red')
    ax2.set_ylim(8e-7, 1.2e-6)
    ax2.axhline(1e-6, color='r', linestyle='--')  # Reference line
    ax2.set_yscale('log')

    # Hide top spines
    ax2.spines['top'].set_visible(False)

    ax1.set_title(f"(MAE error: {np.mean(total_errors):.2e})")
    ax2.set_xlabel("Time Step")

    # Y-axis label for entire figure
    fig.text(0.005, 0.5, 'Mean Absolute Error', va='center', rotation='vertical')

    # Add legend
    ax1.legend(loc='upper right')

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.05)

    # Add broken axis markers
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d / 4, +d / 4), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d / 4, +d / 4), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Set specific ticks for lower plot
    ax2.set_yticks([1e-6])
    ax2.set_yticklabels(['$10^{-6}$'])
    ax2.yaxis.set_minor_locator(plt.NullLocator())

    plt.suptitle(f"{experiment_name} {evaluation_mode.title()}", fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "error_b_analysis.png"))
    plt.close()


# 在现有的visualization.py文件末尾添加以下函数

import re
from scipy import stats
from scipy.interpolate import interp1d
import joblib


def plot_grain_kinetics_comparison(
        ml_joblib_path: str,
        pf_joblib_path: str,
        output_filename: str = 'grain_kinetics_comparison.png'
) -> Dict:
    """
    Plot comparative grain growth kinetics with linear fitting.

    Args:
        ml_joblib_path: Path to ML results joblib file.
        pf_joblib_path: Path to PF results joblib file.
        output_filename: Output image filename.

    Returns:
        results: Dictionary containing fitting results.
    """
    # Load ML model joblib data
    ml_results = joblib.load(ml_joblib_path)

    # Load PF simulation joblib data
    pf_results = joblib.load(pf_joblib_path)

    # Process ML model data
    ml_time_values = []
    ml_radius_values = []

    pattern = re.compile(r'Predicted Phi(\d+)')

    for key, data in ml_results.items():
        match = pattern.match(key)
        if match and 'avg_radius' in data and data['actual_grains'] > 0:
            time = int(match.group(1))
            radius = data['avg_radius']
            ml_time_values.append(time)
            ml_radius_values.append(radius)

    # Process PF simulation data
    pf_time_values = pf_results['time_points']
    pf_radius_values = []

    for result in pf_results['results']:
        if result['actual_grains'] > 0:
            pf_radius_values.append(result['avg_radius'])
        else:
            pf_radius_values.append(np.nan)

    # Remove NaN values
    pf_valid_indices = ~np.isnan(pf_radius_values)
    pf_time_array = np.array(pf_time_values)[pf_valid_indices] - 20
    pf_radius_array = np.array(pf_radius_values)[pf_valid_indices]

    if not ml_time_values or len(pf_radius_array) == 0:
        print("Error: No valid data found")
        return {}

    # Convert to arrays and calculate radius squared
    ml_time_array = np.array(ml_time_values)
    ml_radius_array = np.array(ml_radius_values)
    ml_radius_squared = ml_radius_array ** 2
    pf_radius_squared = pf_radius_array ** 2

    # Sort ML data by time
    ml_sort_indices = np.argsort(ml_time_array)
    ml_time_array = ml_time_array[ml_sort_indices]
    ml_radius_squared = ml_radius_squared[ml_sort_indices]

    # Linear regression
    ml_slope, ml_intercept, ml_r_value, ml_p_value, ml_std_err = stats.linregress(
        ml_time_array, ml_radius_squared
    )
    pf_slope, pf_intercept, pf_r_value, pf_p_value, pf_std_err = stats.linregress(
        pf_time_array, pf_radius_squared
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot data points
    ax.scatter(ml_time_array, ml_radius_squared,
               marker='o', color='#1f77b4', edgecolor='black',
               s=30, alpha=0.8, label='ML Simulation')

    ax.scatter(pf_time_array[13:-1], pf_radius_squared[13:-1],
               marker='s', color='#ff7f0e', edgecolor='black',
               s=30, alpha=0.8, label='PF Simulation')

    # Plot fit lines
    ml_fit_line = ml_slope * ml_time_array + ml_intercept
    ax.plot(ml_time_array, ml_fit_line, '-', color='#1f77b4',
            linewidth=2, label='ML Linear Fit')

    pf_time_for_fit = np.linspace(min(pf_time_array), max(pf_time_array), 100)
    pf_fit_line = pf_slope * pf_time_for_fit + pf_intercept
    ax.plot(pf_time_for_fit, pf_fit_line, '-', color='#ff7f0e',
            linewidth=2, label='PF Linear Fit')

    # Add correlation info
    text_str = (f'ML: $corr = {ml_r_value:.4f}$, $slope = {ml_slope:.2f}$\n'
                f'PF: $corr = {pf_r_value:.4f}$, $slope = {pf_slope:.2f}$')

    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    # Set labels and formatting
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel(r'$\langle$R$\rangle^2$ ($\mathrm{μm}^2$)', fontsize=16)
    ax.legend(frameon=True, framealpha=0.9, loc='lower right', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Image saved as: {output_filename}")

    return {
        'ml': {
            'time': ml_time_array,
            'radius_squared': ml_radius_squared,
            'slope': ml_slope,
            'intercept': ml_intercept,
            'r_value': ml_r_value
        },
        'pf': {
            'time': pf_time_array,
            'radius_squared': pf_radius_squared,
            'slope': pf_slope,
            'intercept': pf_intercept,
            'r_value': pf_r_value
        }
    }


# 在现有visualization.py文件中添加以下新函数：
def plot_steady_state_distribution_comparison(
        combined_data_file: str,
        output_filename: str = 'steady_state_distribution_comparison.png'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot steady-state grain size distribution comparison with histogram (ML) and scatter (PF).

    Args:
        combined_data_file: Path to combined steady-state data file.
        output_filename: Output filename.

    Returns:
        ml_bin_centers: ML bin centers.
        ml_distribution: ML distribution.
        std_frequencies: Standard deviations (zeros for this case).
    """
    # Load combined data
    data = joblib.load(combined_data_file)

    ml_data = data['ml_steady_state']
    pf_data = data['pf_steady_state']

    # Extract ML distribution data
    ml_bin_centers = ml_data['bin_centers']
    ml_distribution = ml_data['distribution']
    ml_time_range = ml_data['time_range']

    # Extract PF distribution data
    pf_bin_centers = pf_data['bin_centers']
    pf_distribution = pf_data['distribution']
    pf_time_range = pf_data['time_range']

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot ML data as bars
    bar_width = (ml_bin_centers[1] - ml_bin_centers[0]) * 0.8
    bars = ax.bar(ml_bin_centers, ml_distribution, width=bar_width,
                  alpha=0.7, color='skyblue', edgecolor='black',
                  linewidth=1.0, label='ML Simulation')

    # Plot PF data as scatter points
    ax.scatter(pf_bin_centers, pf_distribution,
               marker='o', color='#2ca02c', s=80, alpha=0.9,
               label='PF Simulation', zorder=10)

    # Set labels and formatting
    ax.set_xlabel(r'Relative Radius R/$\langle$R$\rangle$', fontsize=16)
    ax.set_ylabel('Normalized Frequency', fontsize=16)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=True, framealpha=0.9, fontsize=14)

    # # Add time range information as text
    # info_text = f'ML range: {ml_time_range[0]}-{ml_time_range[1]}\nPF range: {pf_time_range[0]}-{pf_time_range[1]}'
    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Steady-state comparison plot saved as: {output_filename}")

    return ml_bin_centers, ml_distribution, np.zeros_like(ml_distribution)


def plot_distribution_evolution_summary(
        ml_distribution_dir: str,
        pf_distribution_dir: str,
        output_dir: str,
        max_plots: int = 6
) -> None:
    """
    Create summary plots showing evolution of grain size distributions.

    Args:
        ml_distribution_dir: Directory containing ML distribution files.
        pf_distribution_dir: Directory containing PF distribution files.
        output_dir: Output directory for summary plots.
        max_plots: Maximum number of time points to plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    from glob import glob
    # Get ML distribution files
    ml_files = sorted(glob(os.path.join(ml_distribution_dir, "*_distribution.joblib")))
    pf_files = sorted(glob(os.path.join(pf_distribution_dir, "*_distribution.joblib")))

    if not ml_files or not pf_files:
        print("Warning: No distribution files found")
        return

    # Select evenly spaced files for visualization
    ml_selected = [ml_files[i] for i in np.linspace(0, len(ml_files) - 1, max_plots, dtype=int)]
    pf_selected = [pf_files[i] for i in np.linspace(0, len(pf_files) - 1, max_plots, dtype=int)]

    # Create individual comparison plots
    for i, (ml_file, pf_file) in enumerate(zip(ml_selected, pf_selected)):
        # Load data
        ml_data = joblib.load(ml_file)
        pf_data = joblib.load(pf_file)

        # Extract time information
        ml_time = ml_data.get('step_name', f'ML_{i}')
        pf_time = pf_data.get('frame', f'PF_{i}')

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot histograms
        ml_hist = ml_data['histogram']
        pf_hist = pf_data['histogram']

        bar_width = np.diff(ml_hist['bin_centers'])[0] * 0.35

        ax.bar(ml_hist['bin_centers'] - bar_width / 2, ml_hist['counts'],
               width=bar_width, alpha=0.7, color='skyblue',
               edgecolor='black', linewidth=0.5, label=f'ML ({ml_time})')

        ax.bar(pf_hist['bin_centers'] + bar_width / 2, pf_hist['counts'],
               width=bar_width, alpha=0.7, color='lightcoral',
               edgecolor='black', linewidth=0.5, label=f'PF (Frame {pf_time})')

        ax.set_xlabel(r'Relative Radius R/$\langle$R$\rangle$')
        ax.set_ylabel('Normalized Frequency')
        ax.set_title(f'Grain Size Distribution Comparison - Time Point {i + 1}')
        ax.set_xlim(0, 2.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add grain count information
        info_text = f'ML grains: {ml_data["grain_count"]}\nPF grains: {pf_data["grain_count"]}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_timepoint_{i + 1:02d}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create a summary grid plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (ml_file, pf_file) in enumerate(zip(ml_selected, pf_selected)):
        if i >= len(axes):
            break

        ax = axes[i]

        # Load data
        ml_data = joblib.load(ml_file)
        pf_data = joblib.load(pf_file)

        # Plot histograms
        ml_hist = ml_data['histogram']
        pf_hist = pf_data['histogram']

        bar_width = np.diff(ml_hist['bin_centers'])[0] * 0.35

        ax.bar(ml_hist['bin_centers'] - bar_width / 2, ml_hist['counts'],
               width=bar_width, alpha=0.7, color='skyblue',
               edgecolor='black', linewidth=0.3, label='ML')

        ax.bar(pf_hist['bin_centers'] + bar_width / 2, pf_hist['counts'],
               width=bar_width, alpha=0.7, color='lightcoral',
               edgecolor='black', linewidth=0.3, label='PF')

        ax.set_xlim(0, 2.5)
        ax.set_title(f'Time Point {i + 1}', fontsize=12)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=10)

        if i >= 3:  # Bottom row
            ax.set_xlabel(r'Relative Radius R/$\langle$R$\rangle$')
        if i % 3 == 0:  # Left column
            ax.set_ylabel('Normalized Frequency')

    plt.suptitle('Grain Size Distribution Evolution Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evolution_summary_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution evolution summary saved in {output_dir}")


def plot_evolution_process(
        saved_data: List[Tuple[int, np.ndarray]],
        output_dir: str,
        time_indices: Optional[List[int]] = None
) -> None:
    """
    Visualize grain evolution process.

    Args:
        saved_data: List of saved data [(step, phi_data), ...].
        output_dir: Output directory.
        time_indices: Time indices to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)

    if time_indices is None:
        time_indices = range(len(saved_data))

    for i in time_indices:
        if i < len(saved_data):
            step, phi_data = saved_data[i]

            plt.figure(figsize=(8, 8))
            plt.imshow(phi_data, cmap='viridis', origin='lower', vmin=0.6, vmax=1.0)
            plt.colorbar()
            plt.title(f'Grain Evolution - Step {step}')
            plt.xlabel('X')
            plt.ylabel('Y')

            filename = f'evolution_step_{step:04d}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close()

    print(f"Evolution process images saved in {output_dir}")
