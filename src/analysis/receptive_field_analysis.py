"""
Receptive field analysis utilities for neural network models.

This module provides comprehensive tools for analyzing theoretical and effective
receptive fields of neural networks, with gradient computation and visualization capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Dict, Any, Optional
import os
import sys
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def create_bwr_custom_colormap():
    """
    Create a custom blue-white-red colormap.

    Returns:
        LinearSegmentedColormap: Custom colormap with blue-white-red gradient.
    """
    colors = ["blue", "white", "red"]
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


def create_wr_custom_colormap():
    """
    Create a custom white-to-red colormap.

    Returns:
        LinearSegmentedColormap: Custom colormap from white to red.
    """
    colors = ["white", "red"]
    return LinearSegmentedColormap.from_list("white_red", colors, N=256)


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_tecplot(data: np.ndarray, output_path: str, I: int, J: int) -> None:
    """
    Save data in Tecplot format.

    Args:
        data: 2D array data to save.
        output_path: Output file path.
        I, J: Grid dimensions.
    """
    with open(output_path, 'w') as f:
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        for value in data.flatten():
            f.write(f"{value:.6f}\n")


def compute_theoretical_receptive_field(model: nn.Module) -> int:
    """
    Calculate theoretical receptive field size of the model.

    Args:
        model: Neural network model.

    Returns:
        int: Theoretical receptive field size.
    """
    # Import the specific model class
    try:
        from models.residual_net import ResidualConcatNet, ResidualConcatNet_Res

        if isinstance(model, (ResidualConcatNet, ResidualConcatNet_Res)):
            # Calculate ResNet theoretical receptive field
            num_blocks = model.num_blocks
            kernel_size = model.res_blocks[0][0].conv[1].kernel_size[0]

            # Initial layer receptive field
            receptive_field_size = kernel_size

            # Each residual block contribution
            for _ in range(num_blocks):
                # Each residual block has two convolution layers
                receptive_field_size += 2 * (kernel_size - 1)

            # Final 1x1 convolution doesn't increase receptive field
            return receptive_field_size
        else:
            raise ValueError(f"Unsupported model type: {type(model).__name__}")

    except ImportError:
        raise ImportError("Cannot import model classes for receptive field calculation")


def compute_gradient(model: nn.Module, input_tensor: torch.Tensor, output_point: Tuple[int, int]) -> torch.Tensor:
    """
    Compute gradient of output point with respect to input.

    Args:
        model: Neural network model.
        input_tensor: Input tensor with gradients enabled.
        output_point: (y, x) coordinates of output point.

    Returns:
        torch.Tensor: Gradient tensor.
    """
    # Ensure input requires gradients
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    predicted, _ = model(input_tensor)
    predicted = predicted.squeeze().unsqueeze(0).unsqueeze(0)

    # Select specific output point
    y, x = output_point
    target_output = predicted[0, 0, y, x]

    # Compute gradient
    input_gradient = grad(target_output, input_tensor, retain_graph=True)[0]

    return input_gradient


def calculate_effective_rf_size(grad_map: np.ndarray, threshold: float = 0.05) -> int:
    """
    Calculate effective receptive field size.

    Args:
        grad_map: Gradient magnitude map.
        threshold: Threshold for effective receptive field (relative to max value).

    Returns:
        int: Effective receptive field size (number of pixels).
    """
    # Get maximum value as center point value
    center_val = grad_map.max()

    # Calculate threshold value (percentage of center point value)
    thresh_val = center_val * threshold

    # Count pixels above threshold
    rf_size = np.sum(grad_map > thresh_val)

    return rf_size


def analyze_effective_receptive_field(
        model: nn.Module,
        points: list,
        grad_maps: list,
        point_values: list,
        category: str,
        save_path: str,
        input_image: np.ndarray,
        threshold: float = 0.05
) -> Tuple[float, int, float, list]:
    """
    Analyze effective receptive field for a specific category of points.

    Args:
        model: Neural network model.
        points: List of point coordinates [(y, x), ...].
        grad_maps: List of gradient maps.
        point_values: List of point value information.
        category: Category name.
        save_path: Path to save results.
        input_image: Input image.
        threshold: Threshold for effective receptive field calculation.

    Returns:
        tuple: (avg_rf_diameter, theoretical_rf, ratio, rf_sizes)
    """
    if not grad_maps or not points:
        print(f"Warning: No gradient maps or sampling points for {category} category")
        return None, None, None, []

    # Calculate theoretical receptive field size
    theoretical_rf = compute_theoretical_receptive_field(model)
    rf_radius = theoretical_rf // 2

    # Accumulate receptive fields from all points
    accumulated_rf = np.zeros((theoretical_rf, theoretical_rf))
    rf_sizes = [value[3] for value in point_values]  # Extract rf_size

    # Image dimensions
    h, w = input_image.shape

    for i, ((y, x), grad_normalized) in enumerate(zip(points, grad_maps)):
        # Extract receptive field region
        y_min = max(0, y - rf_radius)
        y_max = min(h, y + rf_radius + 1)
        x_min = max(0, x - rf_radius)
        x_max = min(w, x + rf_radius + 1)

        rf_region = grad_normalized[y_min:y_max, x_min:x_max]

        # Place receptive field region at theoretical receptive field center
        rf_centered = np.zeros((theoretical_rf, theoretical_rf))

        # Calculate offset
        y_offset = rf_radius - (y - y_min)
        x_offset = rf_radius - (x - x_min)

        # Place receptive field region
        y_rf_min = max(0, y_offset)
        y_rf_max = min(theoretical_rf, y_offset + rf_region.shape[0])
        x_rf_min = max(0, x_offset)
        x_rf_max = min(theoretical_rf, x_offset + rf_region.shape[1])

        y_reg_min = max(0, -y_offset)
        y_reg_max = y_reg_min + (y_rf_max - y_rf_min)
        x_reg_min = max(0, -x_offset)
        x_reg_max = x_reg_min + (x_rf_max - x_rf_min)

        rf_centered[y_rf_min:y_rf_max, x_rf_min:x_rf_max] = rf_region[y_reg_min:y_reg_max, x_reg_min:x_reg_max]

        # Accumulate receptive field
        accumulated_rf += rf_centered

    # Average receptive field
    avg_rf = accumulated_rf / len(points)

    # Normalize
    avg_rf_normalized = avg_rf / avg_rf.max()

    # Calculate average effective receptive field size (pixels)
    avg_rf_size_pixels = np.mean(rf_sizes)

    # Calculate effective receptive field radius (assuming circular)
    avg_rf_size = np.sqrt(avg_rf_size_pixels / np.pi)

    # Effective receptive field diameter (equivalent to square side length)
    avg_rf_diameter = avg_rf_size * 2

    # Calculate ratio of effective to theoretical receptive field
    ratio = (avg_rf_diameter / theoretical_rf) * 100

    # Visualize average receptive field
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_rf_normalized, cmap='jet')
    plt.colorbar()

    # Add title with theoretical and effective receptive field information
    plt.title(f"{category} Receptive Field Analysis\n"
              f"Theoretical RF Size: {theoretical_rf}x{theoretical_rf}\n"
              f"Effective RF Diameter: {avg_rf_diameter:.1f}\n"
              f"Effective/Theoretical RF Ratio: {ratio:.2f}%")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{category}_avg_receptive_field.png"), dpi=300)
    plt.close()

    # Save average receptive field data
    np.save(os.path.join(save_path, f"{category}_avg_rf.npy"), avg_rf_normalized)

    return avg_rf_diameter, theoretical_rf, ratio, rf_sizes


def analyze_single_point_receptive_field(
        model: nn.Module,
        input_image: np.ndarray,
        point_info: Tuple[Tuple[int, int], float],
        save_path: str,
        point_idx: int,
        point_category: str
) -> Tuple[np.ndarray, Tuple[int, int], float, float, float, int]:
    """
    Analyze receptive field for a single point and save results.

    Args:
        model: Neural network model.
        input_image: Input image.
        point_info: Point information (coordinate, value).
        save_path: Path to save results.
        point_idx: Point index.
        point_category: Point category.

    Returns:
        tuple: (grad_normalized, coord, input_val, output_val, residual_val, rf_size)
    """
    # Prepare input image
    input_tensor = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0)

    # Get output
    with torch.no_grad():
        output, _ = model(input_tensor)

    output_np = output.squeeze().numpy()
    residual_np = output_np - input_image

    # Calculate gradient
    y, x = point_info[0]
    input_grad = compute_gradient(model, input_tensor, (y, x))

    # Take absolute value of gradient
    grad_magnitude = torch.abs(input_grad[0, 0]).detach().numpy()

    # Normalize gradient to [0,1]
    if grad_magnitude.max() != 0:
        grad_normalized = grad_magnitude / grad_magnitude.max()
    else:
        grad_normalized = grad_magnitude

    # Record input, output and residual values for the point
    input_value = input_image[y, x]
    output_value = output_np[y, x]
    residual_value = residual_np[y, x]

    # Calculate effective receptive field size
    rf_size = calculate_effective_rf_size(grad_normalized)

    return grad_normalized, (y, x), input_value, output_value, residual_value, rf_size


def load_model_for_analysis(model_path: str, model_config: Dict[str, Any]) -> nn.Module:
    """
    Load model for receptive field analysis.

    Args:
        model_path: Path to model file.
        model_config: Model configuration dictionary.

    Returns:
        nn.Module: Loaded model.
    """
    try:
        from src.models.residual_net import ResidualConcatNet

        model = ResidualConcatNet(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    except ImportError:
        raise ImportError("Cannot import model classes")


if __name__ == "__main__":
    pass
