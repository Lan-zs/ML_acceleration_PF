"""
Statistical analysis of receptive fields for different regions in spinodal decomposition.

This script identifies and analyzes receptive fields for different types of regions
in spinodal decomposition images: high concentration (0.9), low concentration (0.2),
and intermediate concentration regions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import h5py
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.receptive_field_analysis import (
    ensure_dir,
    save_tecplot,
    compute_theoretical_receptive_field,
    analyze_single_point_receptive_field,
    analyze_effective_receptive_field
)
from models.residual_net import ResidualConcatNet_Res


def create_bwr_custom_colormap():
    """Create a custom red-blue-white colormap, where 0 is white, -1 is blue, 1 is red"""
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
    return cmap


def sample_points_by_concentration(
        image: np.ndarray,
        max_points: int = 20
) -> Dict[str, List[Tuple[Tuple[int, int], float]]]:
    """
    Sample points from different concentration regions in spinodal decomposition.

    Args:
        image: Input image.
        max_points: Maximum number of sampling points per category.

    Returns:
        dict: Dictionary containing three types of points.
    """
    h, w = image.shape
    margin = 10  # Avoid edge effects
    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[margin:h - margin, margin:w - margin] = True

    # Create masks for different concentration regions
    # High concentration: values close to 0.9
    high_conc_mask = np.logical_and(
        np.isclose(image, 0.945, atol=0.05),
        valid_region
    )

    # Low concentration: values close to 0.2
    low_conc_mask = np.logical_and(
        np.isclose(image, 0.175, atol=0.05),
        valid_region
    )

    # Intermediate concentration: values between 0.3 and 0.8
    intermediate_mask = np.logical_and(
        np.logical_and(image > 0.26, image < 0.86),
        valid_region
    )

    # Get coordinates for each type
    high_conc_coords = np.argwhere(high_conc_mask)
    low_conc_coords = np.argwhere(low_conc_mask)
    intermediate_coords = np.argwhere(intermediate_mask)

    # Sample points
    sampled_points = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    # Sample high concentration points
    if len(high_conc_coords) > 0:
        num_points = min(max_points, len(high_conc_coords))
        indices = np.random.choice(len(high_conc_coords), num_points, replace=False)
        for idx in indices:
            y, x = high_conc_coords[idx]
            sampled_points['high_concentration'].append(((int(y), int(x)), image[y, x]))

    # Sample low concentration points
    if len(low_conc_coords) > 0:
        num_points = min(max_points, len(low_conc_coords))
        indices = np.random.choice(len(low_conc_coords), num_points, replace=False)
        for idx in indices:
            y, x = low_conc_coords[idx]
            sampled_points['low_concentration'].append(((int(y), int(x)), image[y, x]))

    # Sample intermediate concentration points
    if len(intermediate_coords) > 0:
        num_points = min(max_points, len(intermediate_coords))
        indices = np.random.choice(len(intermediate_coords), num_points, replace=False)
        for idx in indices:
            y, x = intermediate_coords[idx]
            sampled_points['intermediate_concentration'].append(((int(y), int(x)), image[y, x]))

    return sampled_points


def visualize_concentration_regions(
        image: np.ndarray,
        save_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Visualize different concentration regions in spinodal decomposition.

    Args:
        image: Input image.
        save_path: Path to save the visualization.

    Returns:
        tuple: (high_conc_mask, low_conc_mask, intermediate_mask)
    """
    h, w = image.shape
    margin = 10
    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[margin:h - margin, margin:w - margin] = True

    # Create masks
    high_conc_mask = np.logical_and(
        np.isclose(image, 0.945, atol=0.05),
        valid_region
    )

    # Low concentration: values close to 0.2
    low_conc_mask = np.logical_and(
        np.isclose(image, 0.175, atol=0.05),
        valid_region
    )

    # Intermediate concentration: values between 0.3 and 0.8
    intermediate_mask = np.logical_and(
        np.logical_and(image > 0.26, image < 0.86),
        valid_region
    )

    # Visualize
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='viridis')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(high_conc_mask, cmap='gray')
    plt.title(f'High Concentration Regions (≈0.945)\nPoints found: {np.sum(high_conc_mask)}')

    plt.subplot(2, 2, 3)
    plt.imshow(low_conc_mask, cmap='gray')
    plt.title(f'Low Concentration Regions (≈0.175)\nPoints found: {np.sum(low_conc_mask)}')

    plt.subplot(2, 2, 4)
    plt.imshow(intermediate_mask, cmap='gray')
    plt.title(f'Intermediate Concentration Regions\nPoints found: {np.sum(intermediate_mask)}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "concentration_regions.png"), dpi=300)
    plt.close()

    return high_conc_mask, low_conc_mask, intermediate_mask


def analyze_receptive_fields_for_image(
        model: nn.Module,
        input_image: np.ndarray,
        image_idx: int,
        save_dir: str
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Analyze receptive fields for all sampling points in one spinodal decomposition image.

    Args:
        model: Neural network model.
        input_image: Input image.
        image_idx: Image index.
        save_dir: Save directory.

    Returns:
        tuple: (rf_stats, all_grad_maps, all_points, all_point_values, rf_sizes_by_category)
    """
    # Create save path
    save_path = os.path.join(save_dir, f"image_{image_idx}")
    ensure_dir(save_path)

    # Visualize concentration regions
    high_conc_mask, low_conc_mask, intermediate_mask = visualize_concentration_regions(
        input_image, save_path
    )

    print(f"High concentration points found: {np.sum(high_conc_mask)}")
    print(f"Low concentration points found: {np.sum(low_conc_mask)}")
    print(f"Intermediate concentration points found: {np.sum(intermediate_mask)}")

    # Sample points
    sampled_points = sample_points_by_concentration(
        input_image,
        max_points=10
    )

    # Store all gradient maps, point coordinates and point value information
    all_grad_maps = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    all_points = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    all_point_values = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    # Analyze receptive field for each category of points
    for category, points in sampled_points.items():
        for i, point_info in enumerate(points):
            point, value = point_info
            grad_normalized, coord, input_val, output_val, residual_val, rf_size = analyze_single_point_receptive_field(
                model, input_image, point_info, save_path, i, category
            )
            all_grad_maps[category].append(grad_normalized)
            all_points[category].append(coord)
            all_point_values[category].append((input_val, output_val, residual_val, rf_size))

    # Prepare input image
    input_tensor = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0)

    # Get output
    with torch.no_grad():
        output, _ = model(input_tensor)

    output_np = output.squeeze().numpy()
    residual_np = output_np - input_image

    # Create visualization showing overall sampling points
    plt.figure(figsize=(16, 12))

    # Input image - mark all sampling points
    plt.subplot(2, 2, 1)
    plt.imshow(input_image, cmap='viridis')
    colors = {'high_concentration': 'red', 'low_concentration': 'blue', 'intermediate_concentration': 'green'}
    for category, coords in all_points.items():
        color = colors[category]
        for y, x in coords:
            plt.plot(x, y, 'o', color=color, markersize=5)
    plt.title('Input with All Sampling Points')
    plt.colorbar()

    # Combined effective receptive field heatmap
    plt.subplot(2, 2, 2)
    # Normalize input image
    min_val, max_val = input_image.min(), input_image.max()
    if max_val > min_val:
        normalized_input = (input_image - min_val) / (max_val - min_val)
    else:
        normalized_input = input_image

    # Combine all gradient maps
    combined_grad = np.zeros_like(input_image)
    total_points = 0
    for category, grad_maps in all_grad_maps.items():
        for grad_map in grad_maps:
            combined_grad += grad_map
            total_points += 1

    # Normalize combined gradient map
    if combined_grad.max() != 0 and total_points > 0:
        combined_grad = combined_grad / total_points
        combined_grad = combined_grad / combined_grad.max()

    plt.imshow(combined_grad, cmap='jet')
    plt.title('Combined Effective Receptive Field')
    plt.colorbar()

    # Residual image
    plt.subplot(2, 2, 3)
    plt.imshow(residual_np, cmap=create_bwr_custom_colormap(),
               vmin=-np.max(np.abs(residual_np)), vmax=np.max(np.abs(residual_np)))
    plt.title('Residual (Output - Input)')
    plt.colorbar()

    # Output image
    plt.subplot(2, 2, 4)
    plt.imshow(output_np, cmap='viridis')
    plt.title('Model Output')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "all_points_analysis.png"), dpi=300)
    plt.close()

    # Analyze effective receptive field for each category of points separately
    rf_stats = {}
    rf_sizes_by_category = {}

    # Combine all points
    all_combined_points = []
    all_combined_grads = []
    all_combined_values = []

    for category in ['high_concentration', 'low_concentration', 'intermediate_concentration']:
        if all_points[category] and all_grad_maps[category]:
            rf_diameter, theoretical_rf, ratio, rf_sizes = analyze_effective_receptive_field(
                model, all_points[category], all_grad_maps[category], all_point_values[category],
                category, save_path, input_image
            )
            rf_stats[category] = (rf_diameter, theoretical_rf, ratio)
            rf_sizes_by_category[category] = rf_sizes

            # Add to combined list
            all_combined_points.extend(all_points[category])
            all_combined_grads.extend(all_grad_maps[category])
            all_combined_values.extend(all_point_values[category])

    # Analyze overall receptive field for all category points
    if all_combined_points and all_combined_grads:
        rf_diameter, theoretical_rf, ratio, _ = analyze_effective_receptive_field(
            model, all_combined_points, all_combined_grads, all_combined_values,
            "All_Concentrations_Combined", save_path, input_image
        )
        rf_stats["All_Concentrations_Combined"] = (rf_diameter, theoretical_rf, ratio)

    # Save statistical results
    with open(os.path.join(save_path, "rf_statistics.txt"), "w") as f:
        f.write("Receptive Field Statistical Results for Spinodal Decomposition:\n")
        f.write("==========================================================\n")
        for category, (diameter, theo_rf, ratio) in rf_stats.items():
            f.write(f"{category} concentration statistical results:\n")
            f.write(f"  Theoretical receptive field size: {theo_rf}x{theo_rf}\n")
            f.write(f"  Effective receptive field diameter: {diameter:.2f}\n")
            f.write(f"  Effective/Theoretical receptive field ratio: {ratio:.2f}%\n\n")

    # Save point values and receptive field sizes
    point_data = []
    for category in ['high_concentration', 'low_concentration', 'intermediate_concentration']:
        for values in all_point_values[category]:
            input_val, output_val, residual_val, rf_size = values
            point_data.append({
                'category': category,
                'input_val': input_val,
                'output_val': output_val,
                'residual_val': residual_val,
                'rf_size': rf_size
            })

    # Save as CSV
    pd.DataFrame(point_data).to_csv(os.path.join(save_path, "point_values_and_rf_sizes.csv"), index=False)

    return rf_stats, all_grad_maps, all_points, all_point_values, rf_sizes_by_category


def load_model(model_path: str, device: str = 'cpu') -> ResidualConcatNet_Res:
    """
    Load trained model for spinodal decomposition.

    Args:
        model_path: Path to model checkpoint.
        device: Device type.

    Returns:
        model: Loaded model.
    """
    model = ResidualConcatNet_Res(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()


    print(f"Model loaded from {model_path}")
    return model


def load_data(h5_path: str) -> np.ndarray:
    """
    Load data from H5 file.

    Args:
        h5_path: Path to H5 file.

    Returns:
        np.ndarray: Phase field data.
    """
    with h5py.File(h5_path, 'r') as f:
        phi_data = f['phi_data'][:]
    return phi_data


def main():
    """Main execution function for spinodal decomposition receptive field analysis."""
    # Configuration paths for spinodal decomposition
    h5_file_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/dataset/spinodal_decomposition_500dt/Training/S375-1_I256_J256_ndt1_mint20_maxt1000.h5"
    model_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/best_checkpoint.pt"  # Update this path
    save_dir = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/effective_receptive_field_analysis"  # Update this path

    # Ensure save directory exists
    ensure_dir(save_dir)

    # Load model
    print("Loading model for spinodal decomposition analysis...")
    model = load_model(model_path, 'cpu')

    # Load data
    print("Loading spinodal decomposition data...")
    data = load_data(h5_file_path)

    # Specify image indices to analyze
    image_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    # Analyze receptive field for each image
    all_image_stats = {}
    all_image_rf_sizes = {}
    all_image_point_values = {}

    # Store points and gradient maps from all images
    all_category_points = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    all_category_grads = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    all_category_values = {
        'high_concentration': [],
        'low_concentration': [],
        'intermediate_concentration': []
    }

    # Save representative image for overall statistics
    representative_image = None

    for idx in image_indices:
        print(f"Analyzing spinodal decomposition image {idx}...")
        if idx < len(data):
            image = data[idx]
            if representative_image is None:
                representative_image = image  # Use first image as representative

            rf_stats, grad_maps, point_coords, point_values, rf_sizes = analyze_receptive_fields_for_image(
                model, image, idx, save_dir
            )
            all_image_stats[idx] = rf_stats
            all_image_rf_sizes[idx] = rf_sizes
            all_image_point_values[idx] = point_values

            # Collect all points and gradient maps for overall average calculation
            for category in ['high_concentration', 'low_concentration', 'intermediate_concentration']:
                all_category_points[category].extend(point_coords[category])
                all_category_grads[category].extend(grad_maps[category])
                all_category_values[category].extend(point_values[category])

    # Analyze receptive field for each category of points combined from all images
    print("Computing overall receptive field statistics for all spinodal decomposition images...")
    overall_rf_stats = {}

    # All categories combined
    all_points = []
    all_grads = []
    all_values = []

    for category in ['high_concentration', 'low_concentration', 'intermediate_concentration']:
        if all_category_points[category] and all_category_grads[category]:
            rf_diameter, theoretical_rf, ratio, _ = analyze_effective_receptive_field(
                model, all_category_points[category], all_category_grads[category], all_category_values[category],
                f"Overall_{category}", save_dir, representative_image
            )
            overall_rf_stats[category] = (rf_diameter, theoretical_rf, ratio)

            # Add to combined list
            all_points.extend(all_category_points[category])
            all_grads.extend(all_category_grads[category])
            all_values.extend(all_category_values[category])

    # Analyze receptive field for all categories combined
    if all_points and all_grads:
        rf_diameter, theoretical_rf, ratio, _ = analyze_effective_receptive_field(
            model, all_points, all_grads, all_values,
            "Overall_All_Concentrations", save_dir, representative_image
        )
        overall_rf_stats["All_Concentrations"] = (rf_diameter, theoretical_rf, ratio)

    # Save overall statistical results
    with open(os.path.join(save_dir, "overall_rf_statistics.txt"), "w") as f:
        f.write("Receptive Field Statistical Results for All Spinodal Decomposition Images:\n")
        f.write("====================================================================\n")
        for category, (diameter, theo_rf, ratio) in overall_rf_stats.items():
            f.write(f"{category} concentration statistical results:\n")
            f.write(f"  Theoretical receptive field size: {theo_rf}x{theo_rf}\n")
            f.write(f"  Effective receptive field diameter: {diameter:.2f}\n")
            f.write(f"  Effective/Theoretical receptive field ratio: {ratio:.2f}%\n\n")

    print(f"Spinodal decomposition receptive field analysis completed! Results saved to {save_dir} directory")


if __name__ == "__main__":
    main()