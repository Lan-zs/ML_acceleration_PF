"""
Statistical analysis of receptive fields for different grain regions.

This script identifies and analyzes receptive fields for different types of regions
in grain growth images: triple junctions, grain boundaries, and grain interiors.
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
from scipy import ndimage
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
from models.residual_net import ResidualConcatNet


def identify_grain_regions(image: np.ndarray) -> np.ndarray:
    """
    Identify grain regions and assign a grain number to each grid point using iterative method.

    Args:
        image: Input image.

    Returns:
        np.ndarray: Labeled image with unique identifier for each grid point.
    """
    # Create grain mask (regions with values > 0.72) and perform initial labeling
    grain_mask = (image > 0.72).astype(int)
    labeled_image, num_features = ndimage.label(grain_mask)

    # Iterative loop until all grid points have labels
    while np.any(labeled_image == 0):
        unlabeled_mask = (labeled_image == 0)
        unlabeled_indices = np.where(unlabeled_mask)

        for i in range(len(unlabeled_indices[0])):
            row, col = unlabeled_indices[0][i], unlabeled_indices[1][i]

            # Extract labels of 8 neighbors
            neighbors = labeled_image[max(0, row - 1):min(labeled_image.shape[0], row + 2),
                        max(0, col - 1):min(labeled_image.shape[1], col + 2)].flatten()

            # Remove 0s (unlabeled pixels) from neighbors
            neighbors = neighbors[neighbors != 0]

            # If there are labeled neighbors
            if len(neighbors) > 0:
                # Find the most frequent label among neighbors
                counts = np.bincount(neighbors)
                most_common_neighbor = np.argmax(counts)

                # Assign label
                labeled_image[row, col] = most_common_neighbor

    return labeled_image


def identify_triple_junctions(labeled_image: np.ndarray) -> np.ndarray:
    """
    Identify triple junction regions.

    Args:
        labeled_image: Grain-labeled image.

    Returns:
        np.ndarray: Triple junction region mask.
    """
    h, w = labeled_image.shape
    triple_junction_mask = np.zeros((h, w), dtype=bool)

    # For each point, check if its 8 neighbors contain 3 or more different grains
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Get all unique grain identifiers in 3x3 window
            neighborhood = labeled_image[i - 1:i + 2, j - 1:j + 2]
            unique_grains = np.unique(neighborhood)

            # If there are 3 or more different grains, mark as triple junction
            if len(unique_grains) >= 3:
                triple_junction_mask[i, j] = True

    return triple_junction_mask


def identify_grain_boundaries(
        image: np.ndarray,
        triple_junction_mask: np.ndarray,
        threshold_low: float = 0.699,
        threshold_high: float = 0.9
) -> np.ndarray:
    """
    Identify grain boundary regions, excluding triple junction areas.

    Args:
        image: Original image.
        triple_junction_mask: Triple junction region mask.
        threshold_low: Lower threshold for grain boundary regions.
        threshold_high: Upper threshold for grain boundary regions.

    Returns:
        np.ndarray: Grain boundary region mask.
    """
    # Create grain boundary mask (values close to 0.7)
    grain_boundary_mask = np.logical_and(
        image >= threshold_low,
        image <= threshold_high
    )

    # Exclude triple junction regions
    grain_boundary_mask = np.logical_and(
        grain_boundary_mask,
        ~triple_junction_mask
    )

    return grain_boundary_mask


def identify_grain_interiors(
        image: np.ndarray,
        triple_junction_mask: np.ndarray,
        grain_boundary_mask: np.ndarray,
        threshold: float = 0.99,
        window_size: int = 9
) -> np.ndarray:
    """
    Identify grain interior regions.

    Args:
        image: Original image.
        triple_junction_mask: Triple junction region mask.
        grain_boundary_mask: Grain boundary region mask.
        threshold: Threshold for grain interior regions.
        window_size: Window size for checking surrounding regions.

    Returns:
        np.ndarray: Grain interior region mask.
    """
    h, w = image.shape
    half_size = window_size // 2
    grain_interior_mask = np.zeros((h, w), dtype=bool)

    # Initial grain interior mask (values close to 1)
    interior_initial = (image > threshold)

    # For each potential interior point, check its window
    for i in range(half_size, h - half_size):
        for j in range(half_size, w - half_size):
            if interior_initial[i, j]:
                window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
                # If all points in window are close to 1
                if np.all(window > threshold):
                    grain_interior_mask[i, j] = True

    # Exclude triple junction and grain boundary regions
    grain_interior_mask = np.logical_and(
        grain_interior_mask,
        ~np.logical_or(triple_junction_mask, grain_boundary_mask)
    )

    return grain_interior_mask


def sample_points_by_region(
        image: np.ndarray,
        triple_junction_mask: np.ndarray,
        grain_boundary_mask: np.ndarray,
        grain_interior_mask: np.ndarray,
        max_points: int = 20
) -> Dict[str, List[Tuple[Tuple[int, int], float]]]:
    """
    Sample points from triple junction, grain boundary and grain interior regions.

    Args:
        image: Original image.
        triple_junction_mask: Triple junction region mask.
        grain_boundary_mask: Grain boundary region mask.
        grain_interior_mask: Grain interior region mask.
        max_points: Maximum number of sampling points per category.

    Returns:
        dict: Dictionary containing three types of points.
    """
    # Create edge avoidance region (avoid points near image edges)
    h, w = image.shape
    margin = 10
    valid_region = np.zeros((h, w), dtype=bool)
    valid_region[margin:h - margin, margin:w - margin] = True

    # Get valid point coordinates for the three types of regions
    triple_junction_coords = np.argwhere(np.logical_and(triple_junction_mask, valid_region))
    grain_boundary_coords = np.argwhere(np.logical_and(grain_boundary_mask, valid_region))
    grain_interior_coords = np.argwhere(np.logical_and(grain_interior_mask, valid_region))

    # Randomly select specified number of points for each type
    sampled_points = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    # Sample triple junction points
    if len(triple_junction_coords) > 0:
        num_points = min(max_points, len(triple_junction_coords))
        indices = np.random.choice(len(triple_junction_coords), num_points, replace=False)
        for idx in indices:
            y, x = triple_junction_coords[idx]
            sampled_points['triple_junction'].append(((int(y), int(x)), image[y, x]))

    # Sample grain boundary points
    if len(grain_boundary_coords) > 0:
        num_points = min(max_points, len(grain_boundary_coords))
        indices = np.random.choice(len(grain_boundary_coords), num_points, replace=False)
        for idx in indices:
            y, x = grain_boundary_coords[idx]
            sampled_points['grain_boundary'].append(((int(y), int(x)), image[y, x]))

    # Sample grain interior points
    if len(grain_interior_coords) > 0:
        num_points = min(max_points, len(grain_interior_coords))
        indices = np.random.choice(len(grain_interior_coords), num_points, replace=False)
        for idx in indices:
            y, x = grain_interior_coords[idx]
            sampled_points['grain_interior'].append(((int(y), int(x)), image[y, x]))

    return sampled_points


def analyze_receptive_fields_for_image(
        model: nn.Module,
        input_image: np.ndarray,
        image_idx: int,
        save_dir: str
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Analyze receptive fields for all sampling points in one image.

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

    # Identify different regions
    labeled_image = identify_grain_regions(input_image)
    triple_junction_mask = identify_triple_junctions(labeled_image)
    grain_boundary_mask = identify_grain_boundaries(input_image, triple_junction_mask)
    grain_interior_mask = identify_grain_interiors(input_image, triple_junction_mask, grain_boundary_mask)

    print(f"Triple junction points found: {np.sum(triple_junction_mask)}")

    # Visualize identified regions
    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(input_image, cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(triple_junction_mask, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.imshow(grain_boundary_mask, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.imshow(grain_interior_mask, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "region_identification.png"), dpi=300)
    plt.close()

    # Sample points
    sampled_points = sample_points_by_region(
        input_image,
        triple_junction_mask,
        grain_boundary_mask,
        grain_interior_mask,
        max_points=10
    )

    # Store all gradient maps, point coordinates and point value information
    all_grad_maps = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    all_points = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    all_point_values = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
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
    for category, coords in all_points.items():
        color = 'red' if category == 'triple_junction' else ('green' if category == 'grain_boundary' else 'blue')
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
    plt.imshow(residual_np, cmap='RdBu_r')
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

    for category in ['triple_junction', 'grain_boundary', 'grain_interior']:
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
            "All_Categories_Combined", save_path, input_image
        )
        rf_stats["All_Categories_Combined"] = (rf_diameter, theoretical_rf, ratio)

    # Save statistical results
    with open(os.path.join(save_path, "rf_statistics.txt"), "w") as f:
        f.write("Receptive Field Statistical Results:\n")
        f.write("==================================\n")
        for category, (diameter, theo_rf, ratio) in rf_stats.items():
            f.write(f"{category} category statistical results:\n")
            f.write(f"  Theoretical receptive field size: {theo_rf}x{theo_rf}\n")
            f.write(f"  Effective receptive field diameter: {diameter:.2f}\n")
            f.write(f"  Effective/Theoretical receptive field ratio: {ratio:.2f}%\n\n")

    # Save point values and receptive field sizes
    point_data = []
    for category in ['triple_junction', 'grain_boundary', 'grain_interior']:
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

def load_model(model_path: str, device: str = 'cuda:0') -> ResidualConcatNet:
    """
    Load trained model.

    Args:
        model_path: Path to model checkpoint.
        device: Device type.

    Returns:
        model: Loaded model.
    """
    model = ResidualConcatNet(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=torch.nn.GELU,
        norm_2d=torch.nn.BatchNorm2d,
        min_value=0.7,
        max_value=1.0
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
    """Main execution function."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    h5_file_path = os.path.join(
        repo_root,
        'dataset',
        'grain_growth',
        'Test',
        '256',
        'Isothermal1023_256256_test_I256_J256_ndt100_mint2000_maxt100000_cleaned75.h5'
    )
    model_path = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'best_checkpoint.pt'
    )
    save_dir = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'effective_receptive_field_analysis'
    )


    # Ensure save directory exists
    ensure_dir(save_dir)

    # Load model
    print("Loading model...")
    model = load_model(model_path,  'cpu')

    # Load data
    print("Loading data...")
    data = load_data(h5_file_path)

    # Specify image indices to analyze
    image_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    # Analyze receptive field for each image
    all_image_stats = {}
    all_image_rf_sizes = {}
    all_image_point_values = {}

    # Store points and gradient maps from all images
    all_category_points = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    all_category_grads = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    all_category_values = {
        'triple_junction': [],
        'grain_boundary': [],
        'grain_interior': []
    }

    # Save representative image for overall statistics
    representative_image = None

    for idx in image_indices:
        print(f"Analyzing image {idx}...")
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
            for category in ['triple_junction', 'grain_boundary', 'grain_interior']:
                all_category_points[category].extend(point_coords[category])
                all_category_grads[category].extend(grad_maps[category])
                all_category_values[category].extend(point_values[category])

    # Analyze receptive field for each category of points combined from all images
    print("Computing overall receptive field statistics for all images...")
    overall_rf_stats = {}

    # All categories combined
    all_points = []
    all_grads = []
    all_values = []

    for category in ['triple_junction', 'grain_boundary', 'grain_interior']:
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
            "Overall_All_Categories", save_dir, representative_image
        )
        overall_rf_stats["All_Categories"] = (rf_diameter, theoretical_rf, ratio)

    # Save overall statistical results
    with open(os.path.join(save_dir, "overall_rf_statistics.txt"), "w") as f:
        f.write("Receptive Field Statistical Results for All Images:\n")
        f.write("================================================\n")
        for category, (diameter, theo_rf, ratio) in overall_rf_stats.items():
            f.write(f"{category} category statistical results:\n")
            f.write(f"  Theoretical receptive field size: {theo_rf}x{theo_rf}\n")
            f.write(f"  Effective receptive field diameter: {diameter:.2f}\n")
            f.write(f"  Effective/Theoretical receptive field ratio: {ratio:.2f}%\n\n")

    print(f"Analysis completed! Results saved to {save_dir} directory")


if __name__ == "__main__":
    main()
