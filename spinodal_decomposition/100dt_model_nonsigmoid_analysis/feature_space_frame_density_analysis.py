"""
Frame density analysis in feature space using a saved reducer.

This script loads a reducer (Only PCA!), samples points on specific frames,
extracts features and projects them to 2D, and renders density contour maps.
"""

import os
import sys
import pickle
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.feature_space_analysis import (
    ensure_dir,
    register_feature_hook,
    create_smooth_nonconvex_boundary,
)
from models.residual_net import ResidualConcatNet


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the ResidualConcatNet as used in the runner."""
    model = ResidualConcatNet(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=torch.nn.GELU,
        norm_2d=torch.nn.BatchNorm2d,
        min_value=0.175,  # For spinodal decomposition
        max_value=0.945   # For spinodal decomposition
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sample_points_in_frame(frame: torch.Tensor, num_samples: int, rf: int) -> np.ndarray:
    """Uniformly sample valid center coordinates within a frame, avoiding borders.

    Returns an array of shape (num_samples, 2) with (y, x) coordinates.
    """
    _, _, H, W = frame.shape
    half = rf // 2
    ys = np.random.randint(half, H - half, size=num_samples)
    xs = np.random.randint(half, W - half, size=num_samples)
    return np.stack([ys, xs], axis=1)


def extract_features_at_points(model: nn.Module, device: torch.device, frame_np: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Run a forward pass to populate hook features and extract feature vectors at coords.

    Returns array of shape (num_points, feature_dim).
    """
    handle, feats = register_feature_hook(model)
    frame = torch.from_numpy(frame_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(frame)
        _ = pred  # silence unused
    fmap = feats['features'][0].cpu().numpy()  # (C, H, W)
    feature_list: List[np.ndarray] = []
    for y, x in coords:
        feature_list.append(fmap[:, int(y), int(x)])
    handle.remove()
    return np.stack(feature_list, axis=0)


def load_reducer(results_file: str):
    """Load saved reducer object from results file."""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    reducer = data.get('reducer', None)
    method = data.get('method', 'pca')
    if reducer is None:
        raise RuntimeError(f"Reducer not found in {results_file}")
    if method.lower() != 'pca':
        print(f"Warning: reducer method is {method}, expected PCA with transform support.")
    return reducer, method


def load_boundary_data(save_dir: str, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load boundary data from saved file."""
    bx, by = np.array([]), np.array([])
    bpath = os.path.join(save_dir, f'{method}_boundary_data.pkl')
    if os.path.exists(bpath):
        with open(bpath, 'rb') as f:
            bdata = pickle.load(f)
            bx, by = bdata.get('boundary_x', np.array([])), bdata.get('boundary_y', np.array([]))
    else:
        print(f"Warning: boundary data file not found at {bpath}")
    return bx, by


def kde_density(coords: np.ndarray, bandwidth: float, grid_size: int,
                global_range: Tuple[float, float, float, float] = None):
    """Compute KDE on 2D coords and return (xx, yy, Z) on specified coordinate grid."""
    if global_range is not None:
        x_min, x_max, y_min, y_max = global_range
    else:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_pad = (x_max - x_min) * 0.1 if x_max > x_min else 0.1
        y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

    xx, yy = np.mgrid[x_min:x_max:grid_size*1j, y_min:y_max:grid_size*1j]

    # Normalize coords for KDE stability
    norm = np.zeros_like(coords)
    norm[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min + 1e-12)
    norm[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min + 1e-12)
    grid_norm = np.vstack([
        (xx.ravel() - x_min) / (x_max - x_min + 1e-12),
        (yy.ravel() - y_min) / (y_max - y_min + 1e-12)
    ]).T

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(norm)
    Z = np.exp(kde.score_samples(grid_norm)).reshape(xx.shape)
    return xx, yy, Z


def plot_density_with_scatter(xx: np.ndarray, yy: np.ndarray, Z: np.ndarray,
                             coords: np.ndarray, boundary: Tuple[np.ndarray, np.ndarray],
                             method: str, frame_idx: int, out_path: str,
                             figsize: Tuple[int, int], dpi: int, levels: int,
                             global_range: Tuple[float, float, float, float] = None):
    """Plot filled contour density with scatter points and dashed boundary."""
    plt.figure(figsize=figsize)

    # Plot density contours
    contour = plt.contourf(xx, yy, Z, levels=levels, cmap='rainbow', alpha=0.85)
    plt.colorbar(contour, label='Density')

    # Plot scatter points
    plt.scatter(coords[:, 0], coords[:, 1],
               color='black', s=30, alpha=0.7, zorder=5, label=f'Frame {frame_idx} samples')

    # Plot boundary if available
    bx, by = boundary
    if bx.size > 0:
        plt.plot(bx, by, '--', color='black', linewidth=3, label='Boundary', zorder=5)

    # Set consistent range if provided
    if global_range is not None:
        x_min, x_max, y_min, y_max = global_range
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.title(f'Density Distribution with Scatter Points - Frame {frame_idx}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def determine_global_range(all_coords_list: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """Determine global range based on all scatter points."""
    all_coords = np.vstack(all_coords_list)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

    # Add padding
    x_pad = (x_max - x_min) * 0.1 if x_max > x_min else 0.1
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    return x_min, x_max, y_min, y_max


def main():
    device = get_device()

    # Paths
    h5_file_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/dataset/spinodal_decomposition_500dt/Training/S375-1_I256_J256_ndt1_mint20_maxt1000.h5"
    model_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_min175/augmentation ScaledResNet_C32_NB8_K3_ndt1/best_checkpoint.pt"  # Update this path
    save_dir = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_min175/augmentation ScaledResNet_C32_NB8_K3_ndt1/feature_space_analysis"  # Update this path

    reduction_results_file = os.path.join(save_dir, 'pca_2d_results.pkl')

    # Frames and sampling
    frame_indices = (0, 500, 900)
    samples_per_frame = 1000
    rf_size = 11

    # Density estimation parameters
    bandwidth = 0.12
    grid_size = 200

    # Plot styling
    figsize = (12, 10)
    dpi = 300
    levels = 60

    ensure_dir(save_dir)

    # Load reducer and boundary
    reducer, method = load_reducer(reduction_results_file)
    bx, by = load_boundary_data(save_dir, method)
    boundary = (bx, by)

    # Load model and data
    with h5py.File(h5_file_path, 'r') as f:
        phi_data = f['phi_data'][:]
    model = load_model(model_path, device)

    # First pass: collect all coordinates to determine global range
    all_coords_list = []
    frame_data_list = []

    for idx in frame_indices:
        frame_np = phi_data[idx]
        frame_tensor = torch.from_numpy(frame_np).float().unsqueeze(0).unsqueeze(0).to(device)
        coords = sample_points_in_frame(frame_tensor, samples_per_frame, rf_size)

        feats = extract_features_at_points(model, device, frame_np, coords)
        reduced = reducer.transform(feats)
        all_coords_list.append(reduced)
        frame_data_list.append((idx, frame_np, coords, feats, reduced))

    # Determine global range
    global_range = determine_global_range(all_coords_list)
    print(f"Global range: x[{global_range[0]:.3f}, {global_range[1]:.3f}], y[{global_range[2]:.3f}, {global_range[3]:.3f}]")

    # Second pass: generate plots with consistent range
    frame_results = []
    for idx, frame_np, coords, feats, reduced in frame_data_list:
        xx, yy, Z = kde_density(reduced, bandwidth=bandwidth, grid_size=grid_size, global_range=global_range)

        # Save density data and boundary
        data = {
            'coords': reduced,
            'xx': xx,
            'yy': yy,
            'Z': Z,
            'boundary_x': bx,
            'boundary_y': by,
            'method': method,
            'frame_index': idx,
            'global_range': global_range,
            'bandwidth': bandwidth,
            'grid_size': grid_size
        }
        data_path = os.path.join(save_dir, f'density_frame_{idx}_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        out_path = os.path.join(save_dir, f'density_frame_{idx}_with_scatter.png')
        plot_density_with_scatter(xx, yy, Z, reduced, boundary, method, idx, out_path, figsize, dpi, levels, global_range)
        frame_results.append((idx, out_path))

    # Save global configuration
    config_data = {
        'frame_indices': frame_indices,
        'samples_per_frame': samples_per_frame,
        'bandwidth': bandwidth,
        'grid_size': grid_size,
        'global_range': global_range,
        'method': method,
        'rf_size': rf_size
    }
    config_path = os.path.join(save_dir, 'density_analysis_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config_data, f)

    print(f"Analysis configuration saved to: {config_path}")
    for idx, path in frame_results:
        print(f"Saved density plot for frame {idx} to: {path}")


if __name__ == "__main__":
    main()