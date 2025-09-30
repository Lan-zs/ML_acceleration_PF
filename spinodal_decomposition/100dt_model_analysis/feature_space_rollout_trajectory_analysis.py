"""
Fixed-point rollout trajectory analysis in reduced feature space.

This script loads a trained model and a saved reducer (e.g., PCA), selects
fixed sampling points, rolls out the model for a given number of frames, and
projects features to visualize trajectories.
"""

import os
import sys
import pickle
from typing import Tuple, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.feature_space_analysis import (
    ensure_dir,
    register_feature_hook,
)
from models.residual_net import ResidualConcatNet_Res


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the ResidualConcatNet as used in the runner."""
    model = ResidualConcatNet_Res(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=torch.nn.GELU,
        norm_2d=torch.nn.BatchNorm2d,
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def select_fixed_sampling_points(frame: torch.Tensor, num_points: int, rf: int) -> List[Tuple[int, int, float]]:
    """Select fixed points within valid region and return (y, x, value)."""
    _, _, H, W = frame.shape
    half = rf // 2
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:half, :] = False
    valid_mask[-half:, :] = False
    valid_mask[:, :half] = False
    valid_mask[:, -half:] = False
    coords = np.argwhere(valid_mask)
    if len(coords) > num_points:
        idx = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[idx]
    points: List[Tuple[int, int, float]] = []
    frame_np = frame.squeeze().cpu().numpy()
    for y, x in coords:
        points.append((int(y), int(x), float(frame_np[y, x])))
    return points


def load_reducer(results_file: str):
    """Load saved reducer object from results file."""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    reducer = data.get('reducer', None)
    method = data.get('method', 'pca')
    if reducer is None:
        raise RuntimeError(f"Reducer not found in {results_file}")
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


def extract_fixed_point_features(model_path: str,
                                 h5_file_path: str,
                                 reducer_file: str,
                                 total_frames: int,
                                 extraction_interval: int,
                                 num_points: int,
                                 rf: int,
                                 save_path: Optional[str]) -> str:
    """Roll out and record reduced feature trajectories for fixed points."""
    ensure_dir(os.path.dirname(save_path) if save_path else '.')
    device = get_device()
    reducer, method = load_reducer(reducer_file)
    model = load_model(model_path, device)
    with h5py.File(h5_file_path, 'r') as f:
        current_frame = torch.from_numpy(f['phi_data'][0:1]).float().unsqueeze(0).to(device)  # [1,1,H,W]

    # Warm-up forward to populate hook structure
    handle, feats = register_feature_hook(model)
    with torch.no_grad():
        _pred, _ = model(current_frame)

    points = select_fixed_sampling_points(current_frame, num_points=num_points, rf=rf)
    trajectories = {
        'sampling_points': points,
        'method': method,
        'reducer': reducer,
        'extraction_interval': extraction_interval,
        'total_frames': total_frames,
        'point_data': [{'point_idx': i,
                        'coordinates': (y, x),
                        'initial_value': v,
                        'frame_indices': [],
                        'reduced_coords': []}
                       for i, (y, x, v) in enumerate(points)]
    }

    with torch.no_grad():
        for t in range(total_frames):
            if t % extraction_interval == 0:
                # Trigger hook for this frame
                _pred, _ = model(current_frame)
                fmap = feats['features'][0].cpu().numpy()
                for i, (y, x, _v) in enumerate(points):
                    feat = fmap[:, y, x].reshape(1, -1)
                    reduced = reducer.transform(feat)[0]
                    trajectories['point_data'][i]['frame_indices'].append(t)
                    trajectories['point_data'][i]['reduced_coords'].append(reduced)
            # Next frame
            next_frame, _ = model(current_frame)
            current_frame = next_frame.squeeze().unsqueeze(0).unsqueeze(0)

    handle.remove()

    if save_path is None:
        save_path = os.path.join('.', 'fixed_points_features.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    return save_path


def plot_fixed_points_evolution(features_file: str,
                                save_dir: str,
                                output_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (13, 12),
                                title_fontsize: int = 16,
                                label_fontsize: int = 40,
                                point_size: int = 100,
                                point_alpha: float = 0.8,
                                tick_size: int = 30,
                                boundary_linewidth: int = 6):
    """Plot fixed-point trajectories over boundary."""
    print(f"Loading features data from: {features_file}")
    with open(features_file, 'rb') as f:
        traj = pickle.load(f)

    # Load boundary data using the same method as in your reference code
    print(f"Loading boundary data from: {save_dir}")

    method = traj.get('method', 'pca')
    bx, by = load_boundary_data(save_dir,method)

    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    fig, ax = plt.subplots(figsize=figsize)

    # Draw boundary if available
    if bx.size > 0 and by.size > 0:
        ax.plot(bx, by, 'k--', linewidth=boundary_linewidth, alpha=0.9)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, pdata in enumerate(traj['point_data']):
        coords = np.array(pdata['reduced_coords'])
        if coords.size == 0:
            continue
        ax.scatter(coords[:, 0], coords[:, 1], color=colors[i % len(colors)], s=point_size, alpha=point_alpha)

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=label_fontsize)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=label_fontsize)

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    # plt.show()
    return fig, ax


def main():
    # Paths and parameters
    h5_file_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/dataset/spinodal_decomposition_500dt/Training/S375-1_I256_J256_ndt1_mint20_maxt1000.h5"
    model_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/best_checkpoint.pt"  # Update this path
    save_dir = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/feature_space_analysis/rollout_fixed_points"  # Update this path

    reduction_results_file = os.path.join(os.path.dirname(save_dir), 'pca_2d_results.pkl')

    total_frames = 500
    extraction_interval = 1
    num_points = 100
    rf_size = 11

    ensure_dir(save_dir)
    # Extract fixed point trajectories
    features_file = extract_fixed_point_features(
        model_path=model_path,
        h5_file_path=h5_file_path,
        reducer_file=reduction_results_file,
        total_frames=total_frames,
        extraction_interval=extraction_interval,
        num_points=num_points,
        rf=rf_size,
        save_path=os.path.join(save_dir, 'fixed_points_features.pkl')
    )

    # Plot trajectories with boundary
    plot_fixed_points_evolution(
        features_file=features_file,
        save_dir=os.path.dirname(save_dir),  # Use the parent directory to find boundary_data.pkl
        output_path=os.path.join(save_dir, 'fixed_points_evolution.png')
    )


if __name__ == "__main__":
    main()