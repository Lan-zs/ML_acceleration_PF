"""
Feature space analysis for non-isothermal grain growth (multi-channel).

This script extracts multi-channel features from H5 datasets using a trained model,
performs PCA-based dimensionality reduction (optionally with cuML acceleration),
and generates visualizations. Structure mirrors the ideal runner while preserving
original logic.
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import random
import pickle
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import seaborn as sns

# Set Seaborn-like style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DeepGrainGrowth_tem')))
try:
    from DeepGrainGrowth_tem.Models.two_Resnet import ResidualConcatNet
    from DeepGrainGrowth_tem.Models.UNetModels import ScaledUNet_not_res
except ImportError:
    print("Failed to import model modules. Please ensure the path is correct.")

# Try to import cuML-accelerated PCA
try:
    from cuml import PCA as cuPCA

    use_cuda = torch.cuda.is_available()
    print("Using CUDA-accelerated PCA (cuML)")
except ImportError:
    use_cuda = False
    print("cuML not installed; falling back to sklearn PCA")


def extract_dataset_name(h5_file_path):
    """Extract dataset name from H5 file path."""
    filename = os.path.basename(h5_file_path)
    dataset_name = filename.split('_')[0]
    return dataset_name


def load_model(checkpoint_path, device, in_channels=1):
    """Load model based on checkpoint path."""
    if 'resnet' in checkpoint_path.lower():
        print(f"Loading ResNet model, in_channels: {in_channels}")
        model = ResidualConcatNet(
            in_channels=in_channels,
            base_channels=96,
            num_blocks=10,
            kernel_size=3,
            act_fn=nn.GELU,
            norm_2d=nn.BatchNorm2d,
            min_value=0.7,
            max_value=1.0
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    return model


def register_hook(model, model_type):
    """Register a forward hook for feature extraction."""
    features = {}

    def hook_fn(module, input, output):
        features['features'] = output.detach()

    if 'resnet' in model_type.lower():
        hook_handle = model.res_blocks[-1][1].register_forward_hook(hook_fn)
    else:
        hook_handle = model.up_path[-1][1].register_forward_hook(hook_fn)

    return hook_handle, features


def extract_features_multi_channel(checkpoint_path, h5_file_path, save_dir='extracted_features', batch_size=32,
                                   sample_range=(0, 5000), is_test=False, num_positions_per_sample=10,
                                   in_channels=2, dataset_name=None):
    """Extract multi-channel features."""
    if dataset_name is None:
        dataset_name = extract_dataset_name(h5_file_path)

    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_prefix = "test" if is_test else "train"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device, in_channels=in_channels)
    model_type = 'resnet' if 'resnet' in checkpoint_path.lower() else 'unet'
    hook_handle, features_dict = register_hook(model, model_type)

    if isinstance(sample_range, tuple):
        start, end = sample_range
        desc = f"Extract training samples [{start}:{end}]"
        file_suffix = f"{start}_{end}"
    else:
        start = sample_range
        end = start + 1
        desc = f"Extract test sample [index:{start}]"
        file_suffix = f"{start}"

    data_file = os.path.join(save_dir, f'{file_prefix}_{dataset_name}_features_{file_suffix}.pkl')

    if os.path.exists(data_file):
        print(f"Loading existing features file: {data_file}")
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            configs_list = data.get('configs_list', [])
            features = data.get('features', [])
            outputs = data.get('outputs', [])
            center_values_list = data.get('center_values_list', [[None] * len(features)] * in_channels)
            print(f"Loaded {len(features)} features")
            return configs_list, features, outputs, center_values_list
        except Exception as e:
            print(f"Failed to load features file: {e}. Re-extracting features.")

    print(f"Reading multi-channel data from {h5_file_path} ...")
    channel_data = []

    with h5py.File(h5_file_path, 'r') as f:
        phi_data = f['phi_data'][start:end]
        channel_data.append(phi_data)

        if in_channels > 1:
            Q = 1.82
            kev = 8.625e-5
            Tmax = 1073.0
            temp_data = f['temp_data'][:]
            temp_data = np.exp(-Q / (kev * temp_data)) / np.exp(-Q / (kev * Tmax))
            repeated_temp_data = np.repeat(temp_data[np.newaxis, :, :], phi_data.shape[0], axis=0)
            channel_data.append(repeated_temp_data)

    receptive_field_size = 11
    half_size = receptive_field_size // 2

    configs_list = [[] for _ in range(in_channels)]
    all_features = []
    all_outputs = []
    all_center_values = [[] for _ in range(in_channels)]

    def process_batch(batch_indices):
        batch_configs_list = [[] for _ in range(in_channels)]
        batch_features = []
        batch_outputs = []
        batch_center_values_list = [[] for _ in range(in_channels)]

        with torch.no_grad():
            for i in batch_indices:
                relative_idx = i - start
                if relative_idx >= len(channel_data[0]):
                    print(f"Warning: index {i} (relative {relative_idx}) out of range {len(channel_data[0])}")
                    continue

                input_tensor = torch.zeros(1, in_channels, channel_data[0][relative_idx].shape[0],
                                           channel_data[0][relative_idx].shape[1],
                                           device=device, dtype=torch.float32)

                for c in range(in_channels):
                    if c == 0:
                        input_tensor[0, c] = torch.from_numpy(channel_data[c][relative_idx]).float().to(device)
                    else:
                        if c < len(channel_data):
                            if c == 1 and len(channel_data[c].shape) == 2:
                                input_tensor[0, c] = torch.from_numpy(channel_data[c]).float().to(device)
                            else:
                                input_tensor[0, c] = torch.from_numpy(channel_data[c][relative_idx]).float().to(device)

                H, W = input_tensor.shape[2], input_tensor.shape[3]
                outputs, _ = model(input_tensor)
                outputs = outputs.squeeze().unsqueeze(0).unsqueeze(0)
                feature_maps = features_dict['features'][0].cpu()

                num_points = 1000 if is_test else num_positions_per_sample

                for _ in range(num_points):
                    center_h = random.randint(half_size, H - half_size - 1)
                    center_w = random.randint(half_size, W - half_size - 1)

                    for c in range(in_channels):
                        config = input_tensor[0, c,
                                 center_h - half_size:center_h + half_size + 1,
                                 center_w - half_size:center_w + half_size + 1].cpu().numpy()
                        center_value = input_tensor[0, c, center_h, center_w].cpu().item()
                        batch_configs_list[c].append(config)
                        batch_center_values_list[c].append(center_value)

                    feature = feature_maps[:, center_h, center_w].numpy()
                    output_value = outputs[0, 0, center_h, center_w].cpu().item()
                    batch_features.append(feature)
                    batch_outputs.append(output_value)

        return batch_configs_list, batch_features, batch_outputs, batch_center_values_list

    data_length = end - start
    batch_indices_list = [range(start + i, min(start + i + batch_size, end)) for i in range(0, data_length, batch_size)]

    for batch_indices in tqdm(batch_indices_list, desc=desc):
        if len(batch_indices) > 0:
            batch_configs_list, batch_features, batch_outputs, batch_center_values_list = process_batch(batch_indices)
            for c in range(in_channels):
                configs_list[c].extend(batch_configs_list[c])
                all_center_values[c].extend(batch_center_values_list[c])
            all_features.extend(batch_features)
            all_outputs.extend(batch_outputs)

    hook_handle.remove()

    with open(data_file, 'wb') as f:
        pickle.dump({
            'configs_list': configs_list,
            'features': all_features,
            'outputs': all_outputs,
            'center_values_list': all_center_values,
            'sample_range': sample_range,
            'in_channels': in_channels,
            'dataset_name': dataset_name
        }, f)

    print(f"Extracted and saved {len(all_features)} features to: {data_file}")
    return configs_list, all_features, all_outputs, all_center_values


def load_features_multi_channel(features_file):
    """Load multi-channel features data."""
    print(f"Loading features data: {features_file}")
    with open(features_file, 'rb') as f:
        data = pickle.load(f)

    configs_list = data['configs_list']
    features = data['features']
    outputs = data['outputs']
    center_values_list = data.get('center_values_list', [[None] * len(features)] * len(configs_list))
    in_channels = data.get('in_channels', len(configs_list))
    dataset_name = data.get('dataset_name', extract_dataset_name(features_file))

    print(f"Loaded {len(features)} features, in_channels: {in_channels}")
    return configs_list, features, outputs, center_values_list, in_channels, dataset_name


def perform_pca_reduction(features, n_components=2):
    """Perform PCA reduction."""
    print(f"Running PCA to {n_components}D ...")
    feature_array = np.array(features)

    if use_cuda:
        reducer = cuPCA(n_components=n_components)
    else:
        reducer = PCA(n_components=n_components)

    reduced_data = reducer.fit_transform(feature_array)

    # Print explained variance ratio if available
    if hasattr(reducer, 'explained_variance_ratio_'):
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")

    return reduced_data, reducer


def calculate_avg_temperature(temp_configs):
    """Compute average temperature per config and convert back to Kelvin."""
    Q = 1.82
    kev = 8.625e-5
    Tmax = 1073.0

    avg_normalized_values = [np.mean(config) for config in temp_configs]

    # Inverse transform from normalized mobility back to temperature (Kelvin)
    # normalized = exp(-Q/(kev*T)) / exp(-Q/(kev*Tmax))
    # Inversion: T = -Q / (kev * ln(normalized * exp(-Q/(kev*Tmax))))
    avg_temps = []
    for norm_val in avg_normalized_values:
        if norm_val > 0:
            temp = -Q / (kev * np.log(norm_val * np.exp(-Q / (kev * Tmax))))
        else:
            temp = Tmax  # Clamp to Tmax if non-positive
        avg_temps.append(temp)

    return np.array(avg_temps)


def normalize_configs_globally(configs_list_dict, channel_idx=0):
    """Global normalization across datasets for a given channel."""
    global_min = float('inf')
    global_max = float('-inf')

    for dataset_name, configs_lists in configs_list_dict.items():
        configs = configs_lists[channel_idx]
        for config in configs:
            min_val = np.min(config)
            max_val = np.max(config)
            global_min = min(global_min, min_val)
            global_max = max(global_max, max_val)

    print(f"Global config range for channel {channel_idx}: [{global_min:.4f}, {global_max:.4f}]")

    normalized_configs_dict = {}
    for dataset_name, configs_lists in configs_list_dict.items():
        configs = configs_lists[channel_idx]
        normalized_configs = []
        for config in configs:
            if global_min != global_max:
                norm_config = (config - global_min) / (global_max - global_min)
            else:
                norm_config = np.ones_like(config) * 0.5
            normalized_configs.append(norm_config)
        normalized_configs_dict[dataset_name] = normalized_configs

    return normalized_configs_dict, global_min, global_max


def create_rgb_config_image(config, cmap_name='viridis'):
    """Convert a normalized config to 3-channel RGB image."""
    cmap = plt.get_cmap(cmap_name)
    rgb_image = cmap(config)
    rgb_image = rgb_image[:, :, :3]
    return rgb_image


def farthest_point_sampling(coords, n_samples=100):
    """Farthest point sampling for uniform spatial sampling.

    Args:
    - coords: array of shape (n_points, n_dimensions)
    - n_samples: number of samples to select

    Returns:
    - indices of sampled points
    """
    from sklearn.preprocessing import StandardScaler

    # Ensure sample count does not exceed available points
    n_points = len(coords)
    n_samples = min(n_samples, n_points)

    if n_samples == n_points:
        return np.arange(n_points)

    # Normalize coordinates
    scaler = StandardScaler()
    normalized_coords = scaler.fit_transform(coords)

    # Initialize from a boundary point (max on first dimension)
    selected_indices = [np.argmax(normalized_coords[:, 0])]

    # Initialize distance tracker
    min_distances = np.inf * np.ones(n_points)

    # Iteratively select remaining points
    for i in range(1, n_samples):
        last_selected = selected_indices[-1]

        # Compute distance to last selected point
        new_distances = np.sum((normalized_coords - normalized_coords[last_selected]) ** 2, axis=1)

        # Update min distances
        min_distances = np.minimum(min_distances, new_distances)

        # Mask selected points to avoid reselection
        min_distances[selected_indices] = -1

        # Select the point with largest minimum distance
        next_point = np.argmax(min_distances)
        selected_indices.append(next_point)

    return np.array(selected_indices)


def plot_phase_field_configs(datasets_coords_dict, normalized_configs_dict,
                             save_path, num_samples=100):
    """Plot phase-field configurations using FPS with scatter and thumbnail images.

    Args:
    - datasets_coords_dict: dict of dataset -> coordinates
    - normalized_configs_dict: dict of dataset -> normalized configs
    - save_path: output image path
    - num_samples: samples per dataset
    """
    fig, ax = plt.subplots(figsize=(18, 14))

    # Merge all datasets
    all_coords = []
    all_configs = []
    all_outputs = []
    dataset_indices = []  # track dataset index per point

    for i, (dataset_name, coords) in enumerate(datasets_coords_dict.items()):
        configs = normalized_configs_dict[dataset_name]
        all_coords.extend(coords)
        all_configs.extend(configs)
        dataset_indices.extend([i] * len(coords))

    all_coords = np.array(all_coords)


    # Plot all points
    scatter = ax.scatter(
        all_coords[:, 0], all_coords[:, 1],
        c='w',
        s=1,
        alpha=1
    )


    # Select points to visualize via FPS
    sample_indices = farthest_point_sampling(all_coords, num_samples)

    print(f"Selected {len(sample_indices)} points for configuration visualization using farthest point sampling")

    # Add configuration thumbnails at sampled points
    for idx in sample_indices:
        x, y = all_coords[idx]
        config = all_configs[idx]

        # Convert to RGB image
        rgb_image = create_rgb_config_image(config)

        # Create image object
        imagebox = OffsetImage(rgb_image, zoom=2.0)

        # Add annotation box with black border
        ab = AnnotationBbox(
            imagebox, (x, y),
            frameon=True,
            pad=0.1,
            bboxprops=dict(edgecolor='black', linewidth=1.5)
        )
        ax.add_artist(ab)

    # Labels
    ax.set_xlabel('PCA Dimension 1', fontsize=45)
    ax.set_ylabel('PCA Dimension 2', fontsize=45)
    ax.tick_params(labelsize=30)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase-field configuration figure saved to: {save_path}")
    plt.close()


def plot_temperature_gradient(coords, temp_configs, save_path):
    """Plot coolwarm scatter colored by average temperature."""
    fig, ax = plt.subplots(figsize=(16, 12))

    avg_temp_values = calculate_avg_temperature(temp_configs)

    print(f"Temperature range: [{np.min(avg_temp_values):.2f}K, {np.max(avg_temp_values):.2f}K]")

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=avg_temp_values,
        cmap='coolwarm',
        s=40,
        alpha=0.7,
        vmin=763,
        vmax=1093
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Average Temperature (K)', fontsize=35, labelpad=15)
    cbar.ax.tick_params(labelsize=25)

    plt.title('PCA Visualization with Temperature Gradient', fontsize=18)
    plt.xlabel('PCA Dimension 1', fontsize=14)
    plt.ylabel('PCA Dimension 2', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Temperature gradient figure saved to: {save_path}")
    plt.close()


def plot_prediction_values(coords, outputs, save_path):
    """Plot viridis scatter colored by predicted values."""
    fig, ax = plt.subplots(figsize=(16, 12))

    outputs_array = np.array(outputs)
    vmin = np.percentile(outputs_array, 5)
    vmax = np.percentile(outputs_array, 95)

    print(f"Output values range: [{np.min(outputs_array):.4f}, {np.max(outputs_array):.4f}]")
    print(f"Using visualization range: [{vmin:.4f}, {vmax:.4f}]")

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=outputs_array,
        cmap='viridis',
        s=30,
        alpha=1,
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Values', fontsize=35, labelpad=15)
    cbar.ax.tick_params(labelsize=25)

    plt.title('PCA Visualization with Prediction Values', fontsize=18)
    plt.xlabel('PCA Dimension 1', fontsize=14)
    plt.ylabel('PCA Dimension 2', fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction values figure saved to: {save_path}")
    plt.close()


def run_pca_analysis_and_visualization(save_dir, in_channels=2):
    """Run PCA and generate three figures."""
    save_dir = os.path.abspath(save_dir)

    train_files = glob.glob(os.path.join(save_dir, 'train_*_features_*.pkl'))
    test_files = glob.glob(os.path.join(save_dir, 'test_*_features_*.pkl'))
    all_files = train_files + test_files

    if not all_files:
        print(f"Error: no feature files found in {save_dir}")
        return

    print(f"Found {len(all_files)} feature files (train: {len(train_files)}, test: {len(test_files)})")

    all_features = []
    all_configs_lists = [[] for _ in range(in_channels)]
    all_outputs = []
    all_center_values_lists = [[] for _ in range(in_channels)]
    all_dataset_labels = []
    all_is_test = []

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        is_test = file_name.startswith('test_')

        if is_test:
            dataset_name = file_name.replace('test_', '').split('_features_')[0]
        else:
            dataset_name = file_name.replace('train_', '').split('_features_')[0]

        configs_list, features, outputs, center_values_list, channels, _ = load_features_multi_channel(file_path)

        feature_count = len(features)
        all_features.extend(features)
        all_outputs.extend(outputs)

        for c in range(in_channels):
            if c < len(configs_list):
                all_configs_lists[c].extend(configs_list[c])
            if c < len(center_values_list):
                all_center_values_lists[c].extend(center_values_list[c])

        all_dataset_labels.extend([dataset_name] * feature_count)
        all_is_test.extend([is_test] * feature_count)

    print(f"Collected {len(all_features)} features for PCA")

    # PCA降维
    pca_results_file = os.path.join(save_dir, 'combined_pca_2d_results.pkl')

    if os.path.exists(pca_results_file):
        print("Loading existing PCA reduction results")
        with open(pca_results_file, 'rb') as f:
            combined_data = pickle.load(f)
        all_reduced = combined_data['coords']
        reducer = combined_data.get('reducer')
    else:
        print("Running new PCA reduction")
        all_reduced, reducer = perform_pca_reduction(all_features, n_components=2)

        with open(pca_results_file, 'wb') as f:
            pickle.dump({
                'coords': all_reduced,
                'configs_lists': all_configs_lists,
                'outputs': all_outputs,
                'center_values_lists': all_center_values_lists,
                'dataset_labels': all_dataset_labels,
                'is_test': all_is_test,
                'method': 'pca',
                'n_components': 2,
                'in_channels': in_channels,
                'reducer': reducer
            }, f)

        print(f"Saved PCA reduction to: {pca_results_file}")

    # 按数据集分组
    datasets_coords_dict = {}
    datasets_outputs_dict = {}
    datasets_configs_list_dict = {}
    datasets_center_values_dict = {}

    unique_datasets = set(all_dataset_labels)
    for dataset in unique_datasets:
        indices = [i for i, label in enumerate(all_dataset_labels) if label == dataset]
        datasets_coords_dict[dataset] = all_reduced[indices]
        datasets_outputs_dict[dataset] = [all_outputs[i] for i in indices]

        dataset_configs_list = []
        dataset_center_values_list = []
        for c in range(in_channels):
            dataset_configs_list.append([all_configs_lists[c][i] for i in indices])
            dataset_center_values_list.append([all_center_values_lists[c][i] for i in indices])

        datasets_configs_list_dict[dataset] = dataset_configs_list
        datasets_center_values_dict[dataset] = dataset_center_values_list

    # Create output directory
    pca_save_dir = os.path.join(save_dir, 'pca')
    os.makedirs(pca_save_dir, exist_ok=True)

    # Fig 1: Phase-field configurations
    print("\nPlotting Fig 1: phase-field configurations ...")
    normalized_configs_dict, min_val, max_val = normalize_configs_globally(
        datasets_configs_list_dict, channel_idx=0
    )
    plot_phase_field_configs(
        datasets_coords_dict,
        normalized_configs_dict,
        os.path.join(pca_save_dir, 'phase_field_configs.png'),
        num_samples=250
    )

    # Fig 2: Temperature gradient scatter
    print("\nPlotting Fig 2: temperature gradient scatter ...")
    all_temp_configs = []
    for dataset in datasets_configs_list_dict:
        all_temp_configs.extend(datasets_configs_list_dict[dataset][1])  # channel 1 is temperature

    plot_temperature_gradient(
        all_reduced,
        all_temp_configs,
        os.path.join(pca_save_dir, 'temperature_gradient.png')
    )

    # Fig 3: Prediction values scatter
    print("\nPlotting Fig 3: prediction values scatter ...")
    plot_prediction_values(
        all_reduced,
        all_outputs,
        os.path.join(pca_save_dir, 'prediction_values.png')
    )

    print("\nAll visualizations completed!")


def extract_and_analyze_multi_channel(checkpoint_path, model_evaluation_H5_folder, save_dir,
                                      train_range=(0, 5000), test_indices=None, in_channels=2):
    """Full multi-channel feature extraction and analysis workflow."""
    if test_indices is None:
        test_indices = [8000]

    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    h5_pattern = os.path.join(model_evaluation_H5_folder, "*.h5")
    h5_files = glob.glob(h5_pattern)

    if not h5_files:
        print(f"Error: no H5 files found in {model_evaluation_H5_folder}")
        return

    print(f"Found {len(h5_files)} H5 files: {[os.path.basename(f) for f in h5_files]}")

    for h5_file_path in h5_files:
        dataset_name = extract_dataset_name(h5_file_path)
        print(f"\n===== Processing dataset: {dataset_name} =====")

        print("\n===== Step 1: Extract training features =====")
        train_file = os.path.join(save_dir, f'train_{dataset_name}_features_{train_range[0]}_{train_range[1]}.pkl')

        if os.path.exists(train_file):
            print(f"Loading existing training features: {train_file}")
        else:
            print("Extracting new training features ...")
            extract_features_multi_channel(
                checkpoint_path, h5_file_path, save_dir,
                sample_range=train_range, is_test=False,
                num_positions_per_sample=5, in_channels=in_channels,
                dataset_name=dataset_name
            )

        print("\n===== Step 2: Extract test features =====")
        for test_idx in test_indices:
            test_file = os.path.join(save_dir, f'test_{dataset_name}_features_{test_idx}.pkl')

            if os.path.exists(test_file):
                print(f"Loading existing test features (index {test_idx}): {test_file}")
            else:
                print(f"Extracting new test features (index {test_idx}) ...")
                extract_features_multi_channel(
                    checkpoint_path, h5_file_path, save_dir,
                    sample_range=test_idx, is_test=True,
                    num_positions_per_sample=1000, in_channels=in_channels,
                    dataset_name=dataset_name
                )

    print("Feature extraction finished. Running PCA and visualization ...")


def main():
    # Paths relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_evaluation_H5_folder = os.path.join(repo_root, 'dataset', 'grain_growth_temp', 'Evaluation')
    checkpoint_path = os.path.join(
        repo_root,
        'non-isothermal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C96_NB10_K3_ndt1',
        'best_checkpoint.pt'
    )
    save_dir = os.path.join(
        repo_root,
        'non-isothermal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C96_NB10_K3_ndt1',
        'features_space'
    )

    # 执行完整的提取和分析流程
    extract_and_analyze_multi_channel(
        checkpoint_path=checkpoint_path,
        model_evaluation_H5_folder=model_evaluation_H5_folder,
        save_dir=save_dir,
        train_range=(0, 500),
        test_indices=[800],
        in_channels=2
    )

    # 执行PCA降维和可视化
    run_pca_analysis_and_visualization(
        save_dir=save_dir,
        in_channels=2
    )


if __name__ == "__main__":
    main()
