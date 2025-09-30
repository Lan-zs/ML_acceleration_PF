"""
Receptive field visualization for specified points in spinodal decomposition images.

This script generates gradient activation visualizations for manually specified points
across different time frames, showing input, receptive field heatmap, and residual images.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.receptive_field_analysis import (
    create_bwr_custom_colormap,
    ensure_dir,
    compute_gradient
)
from models.residual_net import ResidualConcatNet_Res

def load_model(model_path: str, device: str = 'cuda:0') -> ResidualConcatNet_Res:
    """
    Load trained model.

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


def analyze_receptive_field_for_specified_points(
        model: nn.Module,
        input_image: np.ndarray,
        points: list,
        save_path: str
) -> bool:
    """
    Analyze receptive field for specified points and save visualization results.

    Args:
        model: Neural network model.
        input_image: Input image.
        points: List of point coordinates [(y, x), ...].
        save_path: Directory to save results.

    Returns:
        bool: Success status.
    """
    # Prepare input tensor
    input_tensor = torch.from_numpy(input_image).float().unsqueeze(0).unsqueeze(0)

    # Get model output
    with torch.no_grad():
        output, _ = model(input_tensor)

    output_np = output.squeeze().numpy()
    residual_np = output_np - input_image

    # Store gradient information for all points
    all_grad_maps = []

    # Calculate gradient for each point
    for i, point in enumerate(points):
        y, x = point
        input_grad = compute_gradient(model, input_tensor, (y, x))

        # Take absolute value of gradient
        grad_magnitude = torch.abs(input_grad[0, 0]).detach().numpy()

        # Normalize gradient
        if grad_magnitude.max() != 0:
            grad_normalized = grad_magnitude / grad_magnitude.max()
        else:
            grad_normalized = grad_magnitude

        all_grad_maps.append(grad_normalized)

    # Combine all gradient maps
    combined_grad = np.zeros_like(input_image)
    for grad_map in all_grad_maps:
        combined_grad += grad_map

    # Normalize combined gradient map
    if combined_grad.max() != 0:
        combined_grad = combined_grad / combined_grad.max()

    # Save visualization data
    data_to_save = {
        'input_image': input_image,
        'output_image': output_np,
        'residual': residual_np,
        'combined_gradient': combined_grad,
        'points': points,
        'crop_region': (100, 200, 140, 240) # y_min, y_max, x_min, x_max
    }

    np.save(os.path.join(save_path, "visualization_data.npy"), data_to_save)

    # Create visualization with coordinates
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Crop region
    y_min, y_max, x_min, x_max = 100, 200, 140, 240
    image_extent = [x_min, x_max, y_max, y_min]

    # Input image with sampling points
    axes[0].imshow(input_image[y_min:y_max, x_min:x_max], cmap='viridis', extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[0].plot(x, y, 'ro', markersize=6)
            axes[0].text(x + 2, y, f'({x},{y})', color='black', fontsize=9, verticalalignment='center')
    axes[0].set_title('Input with Sampling Points')
    axes[0].set_xlabel('Original X-coordinate')
    axes[0].set_ylabel('Original Y-coordinate')

    # Effective receptive field heatmap
    # Normalize input image
    min_val, max_val = input_image.min(), input_image.max()
    if max_val > min_val:
        normalized_input = (input_image - min_val) / (max_val - min_val)
    else:
        normalized_input = input_image

    # Crop regions
    cropped_input = normalized_input[y_min:y_max, x_min:x_max]
    cropped_grad = combined_grad[y_min:y_max, x_min:x_max]

    # Create heatmap
    heatmap = plt.cm.jet(cropped_grad)[:, :, :3]

    # Blend original image and heatmap
    alpha = 0.7
    blend = (1 - alpha) * np.stack([cropped_input] * 3, axis=-1) + alpha * heatmap
    blend = np.clip(blend, 0, 1)

    axes[1].imshow(blend, extent=image_extent)
    axes[1].set_title('Effective Receptive Field')
    axes[1].set_xlabel('Original X-coordinate')
    axes[1].tick_params(axis='y', labelleft=False)

    # Residual image with sampling points
    axes[2].imshow(residual_np[y_min:y_max, x_min:x_max], cmap=create_bwr_custom_colormap(), extent=image_extent)
    for y, x in points:
        if y_min <= y < y_max and x_min <= x < x_max:
            axes[2].plot(x, y, 'ro', markersize=6)
            axes[2].text(x + 2, y, f'({x},{y})', color='black', fontsize=9, verticalalignment='center')
    axes[2].set_title('Residual (Output - Input)')
    axes[2].set_xlabel('Original X-coordinate')
    axes[2].tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "combined_visualization_with_coords.png"), dpi=300)
    plt.close()

    # Save three separate clean images without any annotations
    # 1. Input image (clean)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(input_image[y_min:y_max, x_min:x_max], cmap='viridis')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "input_clean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2. Receptive field heatmap (clean)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(blend)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "receptive_field_clean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 3. Residual image (clean)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(residual_np[y_min:y_max, x_min:x_max], cmap=create_bwr_custom_colormap())
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "residual_clean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return True


def main():
    """Main execution function."""
    # Configuration
    h5_file_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/dataset/spinodal_decomposition_100dt/Training/S375-1_I256_J256_ndt1_mint1_maxt500.h5"
    model_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/best_checkpoint.pt"  # Update this path
    base_save_dir = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/specified_points_receptive_field_analysis"  # Update this path


    # Frame configurations with specified points
    frame_configs = {
        0:  [(151, 169), (160, 156), (119, 218), (122, 196), (184, 157), (174, 177), (114, 161), (165, 203), (118, 180), (140, 220)],
        10: [(183, 154), (164, 211), (136, 209), (172, 185), (130, 154), (176, 212), (151, 159), (141, 196), (115, 203)],
        148: [(159, 151), (135,161), (150,201), (125,209), (186,196), (130,181),(163,226),(162,177),(176,150),(170,207)],
        480: [(171, 227), (182, 168), (172, 181), (167, 202), (130, 197), (136, 165), (166, 170), (118, 183), (134, 219), (142, 200)]
    }

    # Ensure base save directory exists
    ensure_dir(base_save_dir)

    # Load model
    print("Loading model...")
    model = load_model(model_path, 'cpu')

    # Load data
    print("Loading data...")
    data = load_data(h5_file_path)

    # Analyze each frame
    for image_idx, specified_points in frame_configs.items():
        print(f"Analyzing frame {image_idx}...")

        # Create subdirectory for this frame
        frame_save_dir = os.path.join(base_save_dir, f"frame_{image_idx}")
        ensure_dir(frame_save_dir)

        if image_idx < len(data):
            image = data[image_idx]
            success = analyze_receptive_field_for_specified_points(
                model, image, specified_points, frame_save_dir
            )
            if success:
                print(f"Analysis completed for frame {image_idx}! Results saved to {frame_save_dir}")
            else:
                print(f"Analysis failed for frame {image_idx}")
        else:
            print(f"Error: Frame index {image_idx} exceeds data range")

    print("All frame analyses completed!")


if __name__ == "__main__":
    main()
