"""
Grain Analysis Module for Grain Growth Simulation

This module provides comprehensive tools for analyzing grain growth patterns,
including Voronoi tessellation generation, statistical analysis, and comparison
between deep learning models and phase field simulations.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from scipy.stats import gamma
from skimage.measure import regionprops
from skimage.segmentation import watershed
from matplotlib.colors import LinearSegmentedColormap
import math
import time
from typing import List, Tuple, Optional, Dict, Any
import joblib


def save_tecplot(phi_slice: np.ndarray, output_path: str, I: int, J: int) -> None:
    """
    Save phase field data in Tecplot format.

    Args:
        phi_slice: 2D array of phase field data for single time step.
        output_path: Output file path.
        I, J: Grid dimensions.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('VARIABLE="x","y","FUNCTION"\n')
        f.write(f'ZONE t="BIG ZONE", I={I},  J={J},  F=POINT\n')

        # Write data
        for value in phi_slice.flatten():
            f.write(f"{value:.6f}\n")


def poisson_disk_sampling(width: float, height: float, min_distance: float, k: int = 30) -> np.ndarray:
    """
    Generate Poisson disk sampling points with minimum distance requirement.

    Args:
        width: Domain width.
        height: Domain height.
        min_distance: Minimum distance between points.
        k: Number of attempts before rejection.

    Returns:
        points: Array of sampled points.
    """
    # Initialize grid for accelerated neighbor checking
    cell_size = min_distance / math.sqrt(2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = {}

    points = []
    active_points = []

    # Add first random point
    first_point = np.random.rand(2) * [width, height]
    points.append(first_point)
    active_points.append(first_point)

    # Add point to grid
    grid_x, grid_y = int(first_point[0] / cell_size), int(first_point[1] / cell_size)
    grid[(grid_x, grid_y)] = [first_point]

    # Continue adding points until no active points remain
    while active_points:
        idx = np.random.randint(len(active_points))
        point = active_points[idx]

        # Try to add new points around this point
        found = False
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_distance, 2 * min_distance)
            new_point = point + radius * np.array([np.cos(angle), np.sin(angle)])

            # Handle periodic boundary conditions
            new_point = new_point % [width, height]

            # Check if too close to existing points
            grid_x, grid_y = int(new_point[0] / cell_size), int(new_point[1] / cell_size)
            too_close = False

            # Check surrounding 9 cells
            for i in range(max(0, grid_x - 1), min(grid_width, grid_x + 2)):
                for j in range(max(0, grid_y - 1), min(grid_height, grid_y + 2)):
                    if (i, j) in grid:
                        for p in grid[(i, j)]:
                            # Calculate distance considering periodicity
                            dist_vec = np.abs(new_point - p)
                            dist_vec = np.minimum(dist_vec, [width, height] - dist_vec)
                            dist = np.sqrt(np.sum(dist_vec ** 2))
                            if dist < min_distance:
                                too_close = True
                                break
                if too_close:
                    break

            if not too_close:
                points.append(new_point)
                active_points.append(new_point)
                grid.setdefault((grid_x, grid_y), []).append(new_point)
                found = True
                break

        if not found:
            # Remove current active point if no new point found
            active_points.pop(idx)

    return np.array(points)


def generate_periodic_voronoi(points: np.ndarray, domain_size: float = 1.0, num_replicas: int = 1) -> np.ndarray:
    """
    Generate point set for periodic Voronoi diagram.

    Args:
        points: Original point set.
        domain_size: Domain size.
        num_replicas: Number of replicas for periodic boundaries.

    Returns:
        extended_points: Extended point set including replicas.
    """
    points_list = []

    # Copy original points to surrounding areas for periodic boundary conditions
    for i in range(-num_replicas, num_replicas + 1):
        for j in range(-num_replicas, num_replicas + 1):
            shift = np.array([i, j]) * domain_size
            points_list.append(points + shift)

    # Merge all points
    extended_points = np.vstack(points_list)
    return extended_points


def generate_grain_data(vor: Voronoi, domain_size: float = 1.0, resolution: int = 256) -> np.ndarray:
    """
    Generate grain structure data with grain boundaries at 0.7 and grain interiors at 1.0.

    Args:
        vor: Voronoi diagram object.
        domain_size: Domain size.
        resolution: Grid resolution.

    Returns:
        grid: Grain structure data.
    """
    # Create grid filled with 1.0
    grid = np.ones((resolution, resolution))

    # Create coordinate grid
    x = np.linspace(0, domain_size, resolution)
    y = np.linspace(0, domain_size, resolution)
    X, Y = np.meshgrid(x, y)
    pixel_positions = np.vstack([X.ravel(), Y.ravel()]).T

    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import KDTree
    kdtree = KDTree(vor.points)
    _, region_ids = kdtree.query(pixel_positions)

    # Reshape region IDs to 2D grid
    region_grid = region_ids.reshape(resolution, resolution)

    # Create boundary mask using vectorized operations
    boundary_mask = np.zeros((resolution, resolution), dtype=bool)

    # Horizontal boundaries - compare each pixel with right neighbor
    boundary_mask[:, :-1] |= (region_grid[:, :-1] != region_grid[:, 1:])
    boundary_mask[:, 1:] |= (region_grid[:, :-1] != region_grid[:, 1:])

    # Vertical boundaries - compare each pixel with bottom neighbor
    boundary_mask[:-1, :] |= (region_grid[:-1, :] != region_grid[1:, :])
    boundary_mask[1:, :] |= (region_grid[:-1, :] != region_grid[1:, :])

    # Handle periodic boundaries
    # Left-right boundaries
    boundary_mask[:, 0] |= (region_grid[:, 0] != region_grid[:, -1])
    boundary_mask[:, -1] |= (region_grid[:, 0] != region_grid[:, -1])

    # Top-bottom boundaries
    boundary_mask[0, :] |= (region_grid[0, :] != region_grid[-1, :])
    boundary_mask[-1, :] |= (region_grid[0, :] != region_grid[-1, :])

    # Set boundary pixels to 0.62
    grid[boundary_mask] = 0.62

    return grid


def generate_voronoi_microstructure(
        num_points: int = 40,
        domain_size: float = 1.0,
        resolution: int = 256,
        num_replicas: int = 2,
        seed: int = 42
) -> np.ndarray:
    """
    Generate Voronoi grain structure data.

    Args:
        num_points: Number of grains (seed points).
        domain_size: Domain size.
        resolution: Grid resolution.
        num_replicas: Number of replica regions for periodic boundaries.
        seed: Random seed.

    Returns:
        smoothed_data: Gaussian smoothed grain structure data.
    """
    # Set random seed
    np.random.seed(seed)

    # Use Poisson disk sampling to generate uniformly distributed points
    min_distance = domain_size * 0.8 / math.sqrt(num_points)
    points = poisson_disk_sampling(domain_size, domain_size, min_distance)

    # Adjust min_distance if point count doesn't match expectation
    while len(points) < num_points:
        min_distance *= 0.95  # Slightly decrease minimum distance
        points = poisson_disk_sampling(domain_size, domain_size, min_distance)

    while len(points) > num_points:
        # Randomly remove excess points
        points = points[np.random.choice(len(points), num_points, replace=False)]

    # Create periodic point set
    extended_points = generate_periodic_voronoi(points, domain_size, num_replicas)

    # Calculate Voronoi tessellation
    vor = Voronoi(extended_points)

    # Generate grain structure data
    grain_data = generate_grain_data(vor, domain_size, resolution)

    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(grain_data, sigma=0.8, mode='wrap')

    # Binarize smoothed data
    binary_data = np.where(smoothed_data > 0.99, 1, 0.7)

    # Apply same Gaussian smoothing again
    smoothed_data = gaussian_filter(binary_data, sigma=0.8, mode='wrap')

    return smoothed_data


def model_rollout_with_saving(
        model: torch.nn.Module,
        initial_phi: np.ndarray,
        max_steps: int = 500,
        save_interval: int = 50,
        output_dir: Optional[str] = None
) -> List[Tuple[int, np.ndarray]]:
    """
    Perform long-term rollout using model and save results at specified intervals.

    Args:
        model: Trained model.
        initial_phi: Initial phase field data.
        max_steps: Maximum evolution steps.
        save_interval: Save interval.
        output_dir: Output directory.

    Returns:
        saved_data: List of saved data [(step, phi_data), ...].
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model.eval()
    current_phi = initial_phi.copy()
    saved_data = []
    grid_size = initial_phi.shape[0]

    # Save initial state
    if output_dir:
        save_tecplot(current_phi, os.path.join(output_dir, f'step_0000.dat'), grid_size, grid_size)
    saved_data.append((0, current_phi.copy()))

    print(f"Starting model rollout, total steps: {max_steps}")

    for step in range(max_steps):
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{max_steps}")

        # Prepare model input
        input_tensor = torch.tensor(current_phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Model prediction
        with torch.no_grad():
            predicted_phi_scaled, _ = model(input_tensor)
            predicted_phi = predicted_phi_scaled.squeeze().cpu().numpy()

        # Update current state
        current_phi = predicted_phi

        # Save data at specified intervals
        if (step + 1) % save_interval == 0:
            if output_dir:
                filename = f'step_{step + 1:04d}.dat'
                save_tecplot(current_phi, os.path.join(output_dir, filename), grid_size, grid_size)
            saved_data.append((step + 1, current_phi.copy()))

    print("Model rollout completed")
    return saved_data


def load_dat_file(filepath: str) -> np.ndarray:
    """
    Load phase field data from .dat file.

    Args:
        filepath: Path to .dat file.

    Returns:
        data: Phase field data array.
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Skip header information
        data_start = False
        for line in lines:
            if 'F=POINT' in line:
                data_start = True
                continue
            if data_start:
                try:
                    value = float(line.strip())
                    data.append(value)
                except ValueError:
                    continue

    # Assume data is square grid
    size = int(np.sqrt(len(data)))
    return np.array(data).reshape(size, size)


def identify_grains(phi_field: np.ndarray, threshold: float = 0.85) -> Tuple[List[float], List[float]]:
    """
    Identify grains and calculate grain radii.

    Args:
        phi_field: Phase field data.
        threshold: Grain identification threshold.

    Returns:
        grain_areas: List of grain areas.
        grain_radii: List of grain radii.
    """
    # Binarization
    binary = phi_field > threshold

    # Label connected regions
    labeled, num_features = label(binary)

    # Get region properties
    props = regionprops(labeled)

    grain_areas = []
    grain_radii = []

    for prop in props:
        if prop.area > 10:  # Filter out regions that are too small
            area = prop.area
            # Assume grains are circular, calculate equivalent radius
            radius = np.sqrt(area / np.pi)
            grain_areas.append(area)
            grain_radii.append(radius)

    return grain_areas, grain_radii


def calculate_mean_squared_radius(
        dat_files_or_data: List,
        is_file_list: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate evolution of mean grain radius squared over time.

    Args:
        dat_files_or_data: List of .dat file paths or (step, data) tuples.
        is_file_list: Whether input is file list.

    Returns:
        time_steps: Array of time steps.
        mean_r_squared: Array of mean radius squared values.
    """
    time_steps = []
    mean_r_squared = []

    if is_file_list:
        # Process file list
        for dat_file in sorted(dat_files_or_data):
            # Extract time step from filename
            basename = os.path.basename(dat_file)
            if 'step_' in basename:
                step = int(basename.split('step_')[1].split('.')[0])
            else:
                step = 0

            # Load data
            phi_data = load_dat_file(dat_file)

            # Identify grains and calculate radii
            _, grain_radii = identify_grains(phi_data)

            if grain_radii:
                mean_r_sq = np.mean([r ** 2 for r in grain_radii])
                time_steps.append(step)
                mean_r_squared.append(mean_r_sq)
    else:
        # Process data list
        for step, phi_data in dat_files_or_data:
            # Identify grains and calculate radii
            _, grain_radii = identify_grains(phi_data)

            if grain_radii:
                mean_r_sq = np.mean([r ** 2 for r in grain_radii])
                time_steps.append(step)
                mean_r_squared.append(mean_r_sq)

    return np.array(time_steps), np.array(mean_r_squared)


def calculate_grain_size_distribution(
        phi_data: np.ndarray,
        threshold: float = 0.85,
        normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate grain size distribution.

    Args:
        phi_data: Phase field data.
        threshold: Grain identification threshold.
        normalize: Whether to normalize distribution.

    Returns:
        grain_sizes: Array of grain sizes.
        hist_counts: Histogram counts.
        bin_edges: Histogram bin edges.
    """
    # Identify grains
    _, grain_radii = identify_grains(phi_data, threshold)

    if not grain_radii:
        return np.array([]), np.array([]), np.array([])

    grain_sizes = np.array(grain_radii)

    # Calculate histogram
    n_bins = min(30, len(grain_sizes) // 3 + 1)
    hist_counts, bin_edges = np.histogram(grain_sizes, bins=n_bins, density=normalize)

    return grain_sizes, hist_counts, bin_edges


def analyze_phase_field_data(
        h5_file_path: str,
        save_interval: int = 50,
        max_frames: Optional[int] = None
) -> Dict:
    """
    Analyze phase field simulation data.

    Args:
        h5_file_path: Path to h5 file.
        save_interval: Analysis interval.
        max_frames: Maximum number of frames to analyze.

    Returns:
        analysis_results: Dictionary of analysis results.
    """
    with h5py.File(h5_file_path, 'r') as f:
        phi_data = f['phi_data'][:]

    total_frames = phi_data.shape[0]
    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Select time steps for analysis
    analysis_steps = list(range(0, total_frames, save_interval))
    if analysis_steps[-1] != total_frames - 1:
        analysis_steps.append(total_frames - 1)

    time_steps = []
    mean_r_squared = []
    final_grain_distribution = None

    print(f"Analyzing phase field data, total frames: {total_frames}")

    for i, step in enumerate(analysis_steps):
        if i % 10 == 0:
            print(f"Processing {i + 1}/{len(analysis_steps)} time steps")

        phi_frame = phi_data[step]

        # Calculate grain radii
        _, grain_radii = identify_grains(phi_frame)

        if grain_radii:
            mean_r_sq = np.mean([r ** 2 for r in grain_radii])
            time_steps.append(step)
            mean_r_squared.append(mean_r_sq)

        # Save grain distribution for last frame
        if step == analysis_steps[-1]:
            final_grain_distribution = calculate_grain_size_distribution(phi_frame)

    return {
        'time_steps': np.array(time_steps),
        'mean_r_squared': np.array(mean_r_squared),
        'final_grain_distribution': final_grain_distribution
    }

import matplotlib.pyplot as plt
# 在原有代码基础上，修改和添加以下函数：
def extract_ml_grain_distributions(
        dat_files: List[str],
        output_dir: str
) -> None:
    """
    Extract grain size distributions from ML .dat files and save as histograms.

    Args:
        dat_files: List of .dat file paths.
        output_dir: Output directory for ML distributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Processing ML .dat files...")

    for dat_file in dat_files:
        basename = os.path.basename(dat_file)
        step_name = os.path.splitext(basename)[0]

        print(f"Processing {basename}...")

        # Load phase field data
        phi_data = load_dat_file(dat_file)

        # Identify grains
        _, grain_radii = identify_grains(phi_data)

        if len(grain_radii) > 0:
            # Calculate relative radii
            mean_radius = np.mean(grain_radii)
            relative_radii = np.array(grain_radii) / mean_radius

            # Create histogram
            bins = np.linspace(0, 2.5, 31)  # 30 bins from 0 to 3
            hist_counts, bin_edges = np.histogram(relative_radii, bins=bins, density=True)

            # Save distribution data
            distribution_data = {
                'step_name': step_name,
                'grain_count': len(grain_radii),
                'mean_radius': mean_radius,
                'relative_radii': relative_radii,
                'histogram': {
                    'bin_edges': bin_edges,
                    'counts': hist_counts,
                    'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2
                }
            }

            # Save to individual file
            output_file = os.path.join(output_dir, f"{step_name}_distribution.joblib")
            joblib.dump(distribution_data, output_file)

            # Save histogram plot
            plt.figure(figsize=(8, 6))
            plt.bar(distribution_data['histogram']['bin_centers'],
                    distribution_data['histogram']['counts'],
                    width=np.diff(bin_edges)[0] * 0.8,
                    alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Relative Radius R/<R>')
            plt.ylabel('Normalized Frequency')
            plt.title(f'Grain Size Distribution - {step_name}')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 3)

            plot_file = os.path.join(output_dir, f"{step_name}_histogram.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"ML distributions saved in {output_dir}")


def extract_pf_grain_distributions(
        h5_file_path: str,
        output_dir: str,
        analysis_interval: int = 5
) -> None:
    """
    Extract grain size distributions from PF h5 file and save as histograms.

    Args:
        h5_file_path: Path to h5 file.
        output_dir: Output directory for PF distributions.
        analysis_interval: Interval for analysis (every N frames).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Processing PF h5 file...")

    with h5py.File(h5_file_path, 'r') as f:
        phi_data = f['phi_data'][:]

    total_frames = phi_data.shape[0]
    analysis_steps = list(range(0, total_frames, analysis_interval))

    for step in analysis_steps:
        print(f"Processing frame {step}/{total_frames - 1}...")

        phi_frame = phi_data[step]

        # Identify grains
        _, grain_radii = identify_grains(phi_frame)

        if len(grain_radii) > 0:
            # Calculate relative radii
            mean_radius = np.mean(grain_radii)
            relative_radii = np.array(grain_radii) / mean_radius

            # Create histogram
            bins = np.linspace(0, 2.5, 31)  # 30 bins from 0 to 3
            hist_counts, bin_edges = np.histogram(relative_radii, bins=bins, density=True)

            # Save distribution data
            distribution_data = {
                'frame': step,
                'grain_count': len(grain_radii),
                'mean_radius': mean_radius,
                'relative_radii': relative_radii,
                'histogram': {
                    'bin_edges': bin_edges,
                    'counts': hist_counts,
                    'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2
                }
            }

            # Save to individual file
            output_file = os.path.join(output_dir, f"frame_{step:04d}_distribution.joblib")
            joblib.dump(distribution_data, output_file)

            # Save histogram plot
            plt.figure(figsize=(8, 6))
            plt.bar(distribution_data['histogram']['bin_centers'],
                    distribution_data['histogram']['counts'],
                    width=np.diff(bin_edges)[0] * 0.8,
                    alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('Relative Radius R/<R>')
            plt.ylabel('Normalized Frequency')
            plt.title(f'Grain Size Distribution - Frame {step}')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 3)

            plot_file = os.path.join(output_dir, f"frame_{step:04d}_histogram.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

    print(f"PF distributions saved in {output_dir}")


def calculate_average_steady_state_distribution(
        distribution_dir: str,
        time_range: Tuple[int, int],
        data_type: str = 'ml'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate average steady-state distribution from saved distribution files.

    Args:
        distribution_dir: Directory containing distribution files.
        time_range: Tuple of (start_time, end_time) for steady state.
        data_type: 'ml' or 'pf' to determine file naming pattern.

    Returns:
        bin_centers: Bin centers.
        avg_distribution: Average distribution.
    """
    distribution_files = []

    if data_type == 'ml':
        # Find files in time range for ML data
        for filename in os.listdir(distribution_dir):
            if filename.endswith('_distribution.joblib'):
                # Extract step number from filename
                step_name = filename.replace('_distribution.joblib', '')
                if 'step_' in step_name:
                    try:
                        step_num = int(step_name.split('step_')[1])
                        if time_range[0] <= step_num <= time_range[1]:
                            distribution_files.append(os.path.join(distribution_dir, filename))
                    except ValueError:
                        continue

    elif data_type == 'pf':
        # Find files in time range for PF data
        for filename in os.listdir(distribution_dir):
            if filename.endswith('_distribution.joblib'):
                # Extract frame number from filename
                if 'frame_' in filename:
                    try:
                        frame_num = int(filename.split('frame_')[1].split('_')[0])
                        if time_range[0] <= frame_num <= time_range[1]:
                            distribution_files.append(os.path.join(distribution_dir, filename))
                    except ValueError:
                        continue

    if not distribution_files:
        raise ValueError(f"No distribution files found in time range {time_range}")

    print(f"Found {len(distribution_files)} files in steady state range {time_range}")

    # Load all distributions and average them
    all_distributions = []
    bin_centers = None

    for dist_file in distribution_files:
        data = joblib.load(dist_file)
        histogram = data['histogram']

        if bin_centers is None:
            bin_centers = histogram['bin_centers']

        all_distributions.append(histogram['counts'])

    # Calculate average distribution
    avg_distribution = np.mean(all_distributions, axis=0)

    return bin_centers, avg_distribution


def save_combined_steady_state_data(
        ml_bin_centers: np.ndarray,
        ml_avg_distribution: np.ndarray,
        pf_bin_centers: np.ndarray,
        pf_avg_distribution: np.ndarray,
        ml_time_range: Tuple[int, int],
        pf_time_range: Tuple[int, int],
        output_file: str
) -> None:
    """
    Save combined steady-state distribution data for visualization.

    Args:
        ml_bin_centers: ML bin centers.
        ml_avg_distribution: ML average distribution.
        pf_bin_centers: PF bin centers.
        pf_avg_distribution: PF average distribution.
        ml_time_range: ML time range.
        pf_time_range: PF time range.
        output_file: Output file path.
    """
    combined_data = {
        'ml_steady_state': {
            'bin_centers': ml_bin_centers,
            'distribution': ml_avg_distribution,
            'time_range': ml_time_range
        },
        'pf_steady_state': {
            'bin_centers': pf_bin_centers,
            'distribution': pf_avg_distribution,
            'time_range': pf_time_range
        }
    }

    joblib.dump(combined_data, output_file)
    print(f"Combined steady-state data saved to {output_file}")

