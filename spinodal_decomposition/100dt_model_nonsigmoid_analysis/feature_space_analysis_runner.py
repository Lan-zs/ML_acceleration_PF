"""
Feature space analysis runner for spinodal decomposition 100dt models.

This script performs feature extraction on specified frame ranges, applies
dimensionality reduction (e.g., PCA/TSNE), and generates visualizations. In
line with the receptive field statistics template.
"""

import os
import sys
import pickle
from typing import Tuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch

# Add src to path (mirror receptive_field_imshow.py style)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.feature_space_analysis import (
    ensure_dir,
    extract_features_from_range,
    load_features,
    perform_dimensionality_reduction,
    save_dimensionality_reduction_results,
    load_dimensionality_reduction_results,
    analyze_nearest_neighbor_distances,
    farthest_point_sampling,
    create_smooth_nonconvex_boundary,
)
from models.residual_net import ResidualConcatNet_Res


# ============================
# Helpers for model/data
# ============================

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a ResidualConcatNet (customize here if using a different model)."""
    model = ResidualConcatNet_Res(
        in_channels=1,
        base_channels=32,
        num_blocks=8,
        kernel_size=3,
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_data(h5_path: str) -> np.ndarray:
    """Load phase field frames from an H5 file."""
    with h5py.File(h5_path, 'r') as f:
        phi_data = f['phi_data'][:]
    return phi_data


# ============================
# Plotting helpers
# ============================

def styled_scatter_plot(train_coords: np.ndarray,
                        train_outputs: np.ndarray,
                        test_coords: np.ndarray,
                        test_outputs: np.ndarray,
                        method: str,
                        save_dir: str,
                        filename_prefix: str,
                        boundary_num_harmonics: int = 6,
                        boundary_contour_level: float = 0.00001,
                        boundary_grid_size: int = 100,
                        boundary_bandwidth: float = 0.014) -> str:
    """Create styled 2D scatter with circle markers, colored by output value, and edgecolors by split.

    Saves the boundary points as boundary_data.pkl and returns the plot image path.
    """
    # Set plotting style parameters directly
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16

    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(16, 14))

    # Color range uses both splits
    all_outputs = np.concatenate([train_outputs, test_outputs]) if test_outputs is not None else train_outputs
    vmin, vmax = float(np.min(all_outputs)), float(np.max(all_outputs))

    sc_train = ax.scatter(train_coords[:, 0], train_coords[:, 1], c=train_outputs,
                          cmap='viridis', vmin=vmin, vmax=vmax,
                          s=80, edgecolors='#4878D0', linewidths=1.5, label='Training')

    if test_coords is not None and len(test_coords) > 0:
        ax.scatter(test_coords[:, 0], test_coords[:, 1], c=test_outputs,
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   s=80, edgecolors='#EE854A', linewidths=1.5, label='Test')

    # Boundary from training coordinates only
    try:
        bx, by = create_smooth_nonconvex_boundary(
            train_coords,
            num_harmonics=boundary_num_harmonics,
            contour_level=boundary_contour_level,
            grid_size=boundary_grid_size,
            bandwidth=boundary_bandwidth
        )
        if bx.size > 0:
            ax.plot(bx, by, color='black', linestyle='--', linewidth=3.0, alpha=0.9)
            # Save boundary data for reuse
            with open(os.path.join(save_dir, f'{method}_boundary_data.pkl'), 'wb') as f:
                pickle.dump({'boundary_x': bx, 'boundary_y': by}, f)
    except Exception as e:
        print(f"Boundary creation failed: {e}")

    cbar = plt.colorbar(sc_train)
    cbar.set_label('Predicted Output Value', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=16)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=16)
    ax.legend(fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{filename_prefix}_{method.lower()}_2d_output.png")
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return out_path


def with_configs_plot(train_coords: np.ndarray,
                      train_outputs: np.ndarray,
                      train_configs: List[np.ndarray],
                      test_coords: np.ndarray,
                      test_outputs: np.ndarray,
                      test_configs: List[np.ndarray],
                      method: str,
                      save_dir: str,
                      filename_prefix: str,
                      num_train_configs: int = 60,
                      num_test_configs: int = 15,
                      boundary_contour_level: float = 0.00001,
                      boundary_grid_size: int = 100,
                      boundary_bandwidth: float = 0.014) -> str:
    """Scatter with configuration thumbnails using FPS sampling per split."""
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(26, 20))

    # Color range uses both splits
    all_outputs = np.concatenate([train_outputs, test_outputs]) if test_outputs is not None else train_outputs
    vmin, vmax = float(np.min(all_outputs)), float(np.max(all_outputs))

    sc_train = ax.scatter(train_coords[:, 0], train_coords[:, 1], c=train_outputs,
                          cmap='viridis', vmin=vmin, vmax=vmax,
                          s=80, edgecolors='#4878D0', linewidths=1.5, label='Training')
    if test_coords is not None and len(test_coords) > 0:
        ax.scatter(test_coords[:, 0], test_coords[:, 1], c=test_outputs,
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   s=80, edgecolors='#EE854A', linewidths=1.5, label='Test')

    # Load or create boundary for training points
    bx, by = np.array([]), np.array([])
    bpath = os.path.join(save_dir, 'boundary_data.pkl')
    if os.path.exists(bpath):
        with open(bpath, 'rb') as f:
            bdata = pickle.load(f)
            bx, by = bdata.get('boundary_x', np.array([])), bdata.get('boundary_y', np.array([]))
    if bx.size == 0:
        try:
            bx, by = create_smooth_nonconvex_boundary(
                train_coords,
                contour_level=boundary_contour_level,
                grid_size=boundary_grid_size,
                bandwidth=boundary_bandwidth
            )
        except Exception as e:
            print(f"Boundary creation failed: {e}")
    if bx.size > 0:
        ax.plot(bx, by, color='black', linestyle='--', linewidth=8, alpha=0.7)

    # FPS sampling indices
    t_indices = farthest_point_sampling(train_coords, min(num_train_configs, len(train_coords)))
    v_indices = farthest_point_sampling(test_coords, min(num_test_configs, len(test_coords))) if test_coords is not None and len(test_coords) > 0 else np.array([])

    # Draw train configs
    for idx in t_indices:
        x, y = train_coords[idx]
        config = train_configs[idx]
        # Normalize config to [0.7,1.0] for visibility
        cmin, cmax = float(np.min(config)), float(np.max(config))
        if cmax > cmin:
            config_norm = 0.7 + 0.3 * (config - cmin) / (cmax - cmin)
        else:
            config_norm = np.full_like(config, 0.85)
        img = OffsetImage(config_norm, cmap='viridis', zoom=2.5)
        ab = AnnotationBbox(img, (x, y), frameon=True, pad=0.1, bboxprops=dict(edgecolor='#4878D0', linewidth=8))
        ax.add_artist(ab)

    # Draw test configs
    if v_indices.size > 0:
        for idx in v_indices:
            x, y = test_coords[idx]
            config = test_configs[idx]
            cmin, cmax = float(np.min(config)), float(np.max(config))
            if cmax > cmin:
                config_norm = 0.7 + 0.3 * (config - cmin) / (cmax - cmin)
            else:
                config_norm = np.full_like(config, 0.85)
            img = OffsetImage(config_norm, cmap='viridis', zoom=2.5)
            ab = AnnotationBbox(img, (x, y), frameon=True, pad=0.1, bboxprops=dict(edgecolor='#EE854A', linewidth=8))
            ax.add_artist(ab)

    # Legends and labels
    legend_elements = [
        Patch(facecolor='none', edgecolor='#4878D0', linewidth=8, label='Training'),
        Patch(facecolor='none', edgecolor='#EE854A', linewidth=8, label='Test')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=40)
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=60, labelpad=20)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=60, labelpad=20)
    ax.tick_params(labelsize=40)

    cbar = plt.colorbar(sc_train)
    cbar.set_label('Predicted Output Value', fontsize=40, labelpad=20)
    cbar.ax.tick_params(labelsize=40)

    out_path = os.path.join(save_dir, f"{filename_prefix}_{method.lower()}_2d_with_configs_output.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ============================
# Main workflow
# ============================

def main():
    device = get_device()

    # Paths
    h5_file_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/dataset/spinodal_decomposition_500dt/Training/S375-1_I256_J256_ndt1_mint20_maxt1000.h5"
    model_path = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/best_checkpoint.pt"  # Update this path
    save_dir = "/raid0/zhangzc/zshlan/DeepGrainGrowth/Github_code/spinodal_decomposition/100dt_model_results_nonsigmoid/augmentation ResResNet_C32_NB8_K3_ndt1/feature_space_analysis"  # Update this path


    # Frame ranges (inclusive-exclusive)
    train_range: Tuple[int, int] = (0, 500)
    test_range: Tuple[int, int] = (500, 900)

    # Receptive field size used when extracting per-pixel patches
    rf_size = 11

    # Feature sampling per frame
    num_positions_per_sample_train = 30
    num_positions_per_sample_test = 30

    # Dimensionality reduction settings
    methods = ['pca']  # or ['pca', 'tsne']
    # methods = ['tsne']  # or ['pca', 'tsne']
    n_components_list = [2]
    tsne_perplexity = 30

    # Boundary and sampling parameters
    boundary_num_harmonics = 6
    boundary_contour_level = 0.00001
    boundary_grid_size = 100
    boundary_bandwidth = 0.014
    num_train_configs = 60
    num_test_configs = 15

    ensure_dir(save_dir)

    # Load or compute features for train/test ranges into two files
    train_feat_file = os.path.join(save_dir, f'train_features_{train_range[0]}_{train_range[1]}.pkl')
    test_feat_file = os.path.join(save_dir, f'test_features_{test_range[0]}_{test_range[1]}.pkl')

    if os.path.exists(train_feat_file) and os.path.exists(test_feat_file):
        train_configs, train_features, train_outputs, _ = load_features(train_feat_file)
        test_configs, test_features, test_outputs, _ = load_features(test_feat_file)
    else:
        phi_data = load_data(h5_file_path)
        model = load_model(model_path, device)
        train_configs, train_features, train_outputs, train_centers, train_feat_file = extract_features_from_range(
            model=model,
            device=device,
            phi_data=phi_data,
            save_dir=save_dir,
            sample_range=train_range,
            rf=rf_size,
            num_positions_per_sample=num_positions_per_sample_train,
            label='train'
        )
        test_configs, test_features, test_outputs, test_centers, test_feat_file = extract_features_from_range(
            model=model,
            device=device,
            phi_data=phi_data,
            save_dir=save_dir,
            sample_range=test_range,
            rf=rf_size,
            num_positions_per_sample=num_positions_per_sample_test,
            label='test'
        )

    # Dimensionality reduction (PCA/TSNE), combined results caching
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    combined = np.vstack([X_train, X_test])
    combined_configs = train_configs + test_configs
    combined_outputs = np.concatenate([np.array(train_outputs), np.array(test_outputs)])

    for method in methods:
        for n_components in n_components_list:
            results_file = os.path.join(save_dir, f'{method.lower()}_{n_components}d_results.pkl')
            if os.path.exists(results_file):
                coords, configs_loaded, outputs_loaded, _, _, _ = load_dimensionality_reduction_results(results_file)
            else:
                coords, reducer = perform_dimensionality_reduction(combined, method=method, n_components=n_components, perplexity=tsne_perplexity)
                save_dimensionality_reduction_results(coords, combined_configs, combined_outputs, method, save_dir, center_values=None, reducer=reducer)

            # Split back to train/test using lengths
            train_len = len(train_features)
            train_coords = coords[:train_len]
            test_coords = coords[train_len:]

            # Styled scatter
            styled_scatter_plot(
                train_coords, np.array(train_outputs), test_coords, np.array(test_outputs),
                method, save_dir, filename_prefix='combined',
                boundary_num_harmonics=boundary_num_harmonics,
                boundary_contour_level=boundary_contour_level,
                boundary_grid_size=boundary_grid_size,
                boundary_bandwidth=boundary_bandwidth
            )

            # With configuration thumbnails (FPS sampling)
            with_configs_plot(
                train_coords, np.array(train_outputs), train_configs,
                test_coords, np.array(test_outputs), test_configs,
                method, save_dir, filename_prefix='combined',
                num_train_configs=num_train_configs,
                num_test_configs=num_test_configs,
                boundary_contour_level=boundary_contour_level,
                boundary_grid_size=boundary_grid_size,
                boundary_bandwidth=boundary_bandwidth
            )


if __name__ == "__main__":
    main()