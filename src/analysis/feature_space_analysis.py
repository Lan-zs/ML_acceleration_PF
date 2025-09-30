"""
Feature space analysis utilities.

This module intentionally avoids loading models or datasets so that callers can
inject arbitrary models and data. It focuses on reusable utilities:
- Register forward hooks to capture deep feature maps
- Extract per-pixel features from a given frame range and receptive-field size
- Perform dimensionality reduction (PCA/TSNE only)
- Save/load intermediate features and reduction results
- Compute nearest-neighbor distance distributions
- Provide helper utilities (boundary and sampling) for plotting in runners
"""

import os
import sys
import pickle
import random
from typing import Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KernelDensity
from skimage import measure
import matplotlib.pyplot as plt


# Make src importable if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from analysis.receptive_field_analysis import ensure_dir
# Note: Do not import specific models here; runner will supply the model instance


COLORS = {
    'train': '#4878D0',
    'test': '#EE854A',
    'kde_train': '#2C3E50',
    'kde_test': '#D35400',
    'hist_train_edge': '#2C3E50',
    'hist_test_edge': '#D35400',
    'hist_train_face': '#4878D0',
    'hist_test_face': '#EE854A',
}


def create_bwr_custom_colormap() -> LinearSegmentedColormap:
    colors = ["blue", "white", "red"]
    return LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)


def create_wr_custom_colormap() -> LinearSegmentedColormap:
    colors = ["white", "red"]
    return LinearSegmentedColormap.from_list("white_red", colors, N=256)




def register_feature_hook(model: nn.Module) -> Tuple[Any, dict]:
    """Register a forward hook to capture deep feature maps from a model.

    The caller is responsible for passing a model that exposes a suitable
    final feature block. For the provided ResidualConcatNet, this targets the
    last fusion convolution in the final residual block.
    """
    features: dict = {}

    def hook_fn(module, input, output):
        features['features'] = output.detach()

    handle = model.res_blocks[-1][1].register_forward_hook(hook_fn)
    return handle, features


def extract_features_from_range(
        model: nn.Module,
        device: torch.device,
        phi_data: np.ndarray,
        save_dir: str,
        sample_range: Tuple[int, int],
        rf: int,
        num_positions_per_sample: int,
        label: str,
        batch_size: int = 32
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], str]:
    """Extract features from a contiguous frame range using a registered hook.

    Args:
        model: Trained model in eval mode.
        device: Torch device for inference.
        phi_data: Numpy array of frames; indices are absolute to this array.
        save_dir: Directory to save the extracted feature file.
        sample_range: Inclusive-exclusive range (start, end) of frames to use.
        rf: Receptive field window size (odd integer, e.g., 11).
        num_positions_per_sample: Number of random center positions per frame.
        label: Label for file naming, e.g., 'train' or 'test'.
        batch_size: Number of frames per batch during forward passes.

    Returns:
        configs, features, outputs, center_values, saved_file_path
    """
    ensure_dir(save_dir)
    hook, feats = register_feature_hook(model)

    start, end = sample_range
    half = rf // 2

    all_configs: List[np.ndarray] = []
    all_features: List[np.ndarray] = []
    all_outputs: List[float] = []
    all_centers: List[float] = []

    def process_batch(batch_indices):
        batch_configs, batch_features, batch_outputs, batch_centers = [], [], [], []
        with torch.no_grad():
            for i in batch_indices:
                frame = torch.from_numpy(phi_data[i]).float().unsqueeze(0).unsqueeze(0).to(device)
                H, W = frame.shape[2], frame.shape[3]
                pred, _ = model(frame)
                pred = pred.squeeze().unsqueeze(0).unsqueeze(0)
                feature_maps = feats['features'][0].cpu()
                for _ in range(num_positions_per_sample):
                    cy = random.randint(half, H - half - 1)
                    cx = random.randint(half, W - half - 1)
                    config = frame[0, 0, cy - half:cy + half + 1, cx - half:cx + half + 1].cpu().numpy()
                    center_val = frame[0, 0, cy, cx].cpu().item()
                    feature_vec = feature_maps[:, cy, cx].numpy()
                    output_val = pred[0, 0, cy, cx].cpu().item()
                    batch_configs.append(config)
                    batch_features.append(feature_vec)
                    batch_outputs.append(output_val)
                    batch_centers.append(center_val)
        return batch_configs, batch_features, batch_outputs, batch_centers

    batches = [range(i, min(i + batch_size, end)) for i in range(start, end, batch_size)]
    for batch in tqdm(batches, desc=f"extract {label} [{start}:{end}]"):
        if len(batch) == 0:
            continue
        bc, bf, bo, bcv = process_batch(batch)
        all_configs.extend(bc)
        all_features.extend(bf)
        all_outputs.extend(bo)
        all_centers.extend(bcv)

    hook.remove()

    data_file = os.path.join(save_dir, f"{label}_features_{start}_{end}.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump({
            'configs': all_configs,
            'features': all_features,
            'outputs': all_outputs,
            'center_values': all_centers,
            'sample_range': sample_range
        }, f)

    return all_configs, all_features, all_outputs, all_centers, data_file


def load_features(features_file: str):
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    return data['configs'], data['features'], data['outputs'], data.get('center_values', [None] * len(data['features']))


def perform_dimensionality_reduction(features: List[np.ndarray], method: str = 'pca', n_components: int = 2, perplexity: int = 30, random_state: int = 42):
    X = np.array(features)
    method_l = method.lower()
    if method_l == 'pca':
        reducer = PCA(n_components=n_components)
    elif method_l == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    coords = reducer.fit_transform(X)
    return coords, reducer


def save_dimensionality_reduction_results(reduced_data: np.ndarray, configs: List[np.ndarray], outputs: np.ndarray, method: str, save_dir: str, center_values: Optional[List[float]] = None, reducer: Optional[Any] = None) -> str:
    ensure_dir(save_dir)
    n_components = reduced_data.shape[1]
    results_file = os.path.join(save_dir, f"{method.lower()}_{n_components}d_results.pkl")
    with open(results_file, 'wb') as f:
        payload = {
            'coords': reduced_data,
            'configs': configs,
            'outputs': outputs,
            'method': method,
            'n_components': n_components,
        }
        if center_values is not None:
            payload['center_values'] = center_values
        if reducer is not None:
            payload['reducer'] = reducer
        pickle.dump(payload, f)
    return results_file


def load_dimensionality_reduction_results(results_file: str):
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    return data['coords'], data['configs'], data['outputs'], data['method'], data.get('center_values', [None] * len(data['outputs'])), data.get('reducer', None)


def analyze_nearest_neighbor_distances(train_features: List[np.ndarray], test_features: Optional[List[np.ndarray]] = None, save_dir: Optional[str] = None, filename: str = 'nn_distances.png'):
    if save_dir:
        ensure_dir(save_dir)

    X_train = np.array(train_features)
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    train_distances = distances[:, 1]

    test_distances = None
    if test_features is not None and len(test_features) > 0:
        X_test = np.array(test_features)
        d_test, _ = nn.kneighbors(X_test)
        test_distances = d_test[:, 0]

    if save_dir:
        plt.figure(figsize=(10, 7))
        if test_distances is not None:
            max_dist = max(np.max(train_distances), np.max(test_distances))
        else:
            max_dist = np.max(train_distances)
        bins = np.linspace(0, max_dist * 1.1, 50)
        plt.hist(train_distances, bins=bins, density=True, alpha=0.6, color=COLORS['hist_train_face'], edgecolor=COLORS['hist_train_edge'], label='Training Samples', linewidth=1.5)
        kde_train = stats.gaussian_kde(train_distances)
        x_range = np.linspace(0, max_dist * 1.1, 1000)
        plt.plot(x_range, kde_train(x_range), color=COLORS['kde_train'], linewidth=2.0, label='Training KDE')
        if test_distances is not None:
            plt.hist(test_distances, bins=bins, density=True, alpha=0.6, color=COLORS['hist_test_face'], edgecolor=COLORS['hist_test_edge'], label='Test Samples', linewidth=1.5)
            kde_test = stats.gaussian_kde(test_distances)
            plt.plot(x_range, kde_test(x_range), color=COLORS['kde_test'], linewidth=2.0, label='Test KDE')
        plt.xlabel('Euclidean Distance to NN in Feature Space')
        plt.ylabel('Probability Density')
        plt.title('Nearest Neighbor Distance Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    return train_distances, test_distances


def farthest_point_sampling(coords: np.ndarray, n_samples: int) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    n_points = len(coords)
    n_samples = min(n_samples, n_points)
    if n_samples == n_points:
        return np.arange(n_points)
    scaler = StandardScaler()
    norm_coords = scaler.fit_transform(coords)
    selected = [np.argmax(norm_coords[:, 0])]
    min_d = np.full(n_points, np.inf)
    for _ in range(1, n_samples):
        last = selected[-1]
        new_d = np.sum((norm_coords - norm_coords[last]) ** 2, axis=1)
        min_d = np.minimum(min_d, new_d)
        min_d[selected] = -1
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
    return np.array(selected)


def create_smooth_nonconvex_boundary(points: np.ndarray,num_harmonics: int=7, contour_level: float = 1e-6, grid_size: int = 100, bandwidth: float = 0.014) -> Tuple[np.ndarray, np.ndarray]:
    """Create a smooth non-convex boundary using KDE isocontours with Fourier smoothing.

    Returns arrays boundary_x, boundary_y in the original coordinate system.
    """
    x_orig_min, x_orig_max = points[:, 0].min(), points[:, 0].max()
    y_orig_min, y_orig_max = points[:, 1].min(), points[:, 1].max()

    normalized = np.zeros_like(points)
    for i in range(points.shape[1]):
        min_val = points[:, i].min()
        max_val = points[:, i].max()
        normalized[:, i] = 0.5 if max_val == min_val else (points[:, i] - min_val) / (max_val - min_val)

    x_min, x_max, y_min, y_max = -0.05, 1.05, -0.05, 1.05
    xx, yy = np.mgrid[x_min:x_max:grid_size * 1j, y_min:y_max:grid_size * 1j]
    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(normalized)
    z = np.exp(kde.score_samples(xy_sample)).reshape(xx.shape)
    contours = measure.find_contours(z, contour_level * z.max())
    if len(contours) == 0:
        return np.array([]), np.array([])
    boundary = max(contours, key=len)

    norm_x = x_min + (x_max - x_min) * boundary[:, 1] / grid_size
    norm_y = y_min + (y_max - y_min) * boundary[:, 0] / grid_size

    def fourier_smooth(coords, num_harmonics):
        N = len(coords)
        fft_res = np.fft.fft(coords)
        filt = np.zeros_like(fft_res, dtype=complex)
        filt[0] = fft_res[0]
        for i in range(1, min(num_harmonics + 1, N // 2)):
            filt[i] = fft_res[i]
            filt[N - i] = fft_res[N - i]
        return np.real(np.fft.ifft(filt))

    smooth_x = fourier_smooth(norm_y, num_harmonics)
    smooth_y = fourier_smooth(norm_x, num_harmonics)

    boundary_x = x_orig_min + (x_orig_max - x_orig_min) * smooth_x
    boundary_y = y_orig_min + (y_orig_max - y_orig_min) * smooth_y
    return boundary_x, boundary_y


# Intentionally omit plotting helpers here; the runner will implement styled plots.


def combine_and_reduce(train_features: List[np.ndarray],
                       test_features: List[np.ndarray],
                       train_configs: List[np.ndarray],
                       test_configs: List[np.ndarray],
                       train_outputs: np.ndarray,
                       test_outputs: np.ndarray,
                       save_dir: str,
                       methods: List[str],
                       n_components_list: List[int]) -> None:
    """Combine train/test, run reductions, and save combined results.

    Plotting is intentionally left to the runner so it can style figures.
    """
    ensure_dir(save_dir)
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    combined = np.vstack([X_train, X_test])
    combined_configs = train_configs + test_configs
    combined_outputs = np.concatenate([train_outputs, test_outputs])

    for method in methods:
        for n_components in n_components_list:
            coords, reducer = perform_dimensionality_reduction(combined, method=method, n_components=n_components)
            save_dimensionality_reduction_results(coords, combined_configs, combined_outputs, method, save_dir, center_values=None, reducer=reducer)


# Orchestration is performed by runners; intentionally no top-level workflow here.


