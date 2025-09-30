"""
Grain Growth Kinetics Analysis

This module analyzes grain growth kinetics by comparing deep learning model
predictions with phase field simulations, focusing on mean radius squared evolution.
"""

import os
import sys
import torch
import numpy as np
import h5py
import joblib
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.residual_net import ResidualConcatNet
from analysis.grain_growth_quantitative_validation import (
    generate_voronoi_microstructure,
    model_rollout_with_saving,
    calculate_mean_squared_radius,
    analyze_phase_field_data
)
from utils.visualization import (
    plot_grain_kinetics_comparison,
    plot_evolution_process
)


def load_trained_model(model_path: str, device: str = 'cuda:0') -> ResidualConcatNet:
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


def analyze_grain_kinetics(
        model_path: str,
        phase_field_h5_path: str,
        output_dir: str,
        grain_num: int = 2000,
        grid_size: int = 1024,
        max_steps: int = 500,
        save_interval: int = 50,
        random_seed: int = 42,
        device: str = 'cuda:0'
) -> Dict:
    """
    Analyze grain growth kinetics comparing ML model with phase field simulation.

    Args:
        model_path: Path to trained model.
        phase_field_h5_path: Path to phase field simulation data.
        output_dir: Output directory.
        grain_num: Number of grains.
        grid_size: Grid size.
        max_steps: Maximum evolution steps.
        save_interval: Save interval.
        random_seed: Random seed.
        device: Computation device.

    Returns:
        results: Analysis results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Starting Grain Growth Kinetics Analysis")
    print("=" * 80)

    # 1. Load trained model
    print("1. Loading trained model...")
    model = load_trained_model(model_path, device)

    # 2. Generate Voronoi initial microstructure
    print("2. Generating Voronoi initial microstructure...")
    initial_phi = generate_voronoi_microstructure(
        num_points=grain_num,
        resolution=grid_size,
        seed=random_seed
    )

    # 3. Perform model rollout
    print("3. Performing model rollout...")
    model_output_dir = os.path.join(output_dir, "model_rollout_data")
    saved_model_data = model_rollout_with_saving(
        model=model,
        initial_phi=initial_phi,
        max_steps=max_steps,
        save_interval=save_interval,
        output_dir=model_output_dir
    )

    # 4. Analyze model results
    print("4. Analyzing model evolution results...")
    ml_time_steps, ml_mean_r_squared = calculate_mean_squared_radius(
        saved_model_data, is_file_list=False
    )

    # 5. Analyze phase field simulation data
    print("5. Analyzing phase field simulation data...")
    pf_results = analyze_phase_field_data(
        phase_field_h5_path,
        save_interval=save_interval,
        max_frames=max_steps + 1
    )

    # 6. Perform linear regression analysis
    print("6. Performing linear regression analysis...")

    # ML data regression
    ml_slope, ml_intercept, ml_r_value, ml_p_value, ml_std_err = stats.linregress(
        ml_time_steps, ml_mean_r_squared
    )

    # Phase field data regression
    pf_slope, pf_intercept, pf_r_value, pf_p_value, pf_std_err = stats.linregress(
        pf_results['time_steps'], pf_results['mean_r_squared']
    )

    # 7. Save kinetics data for visualization
    print("7. Saving kinetics data...")

    # Prepare data in the format expected by visualization function
    ml_results_for_viz = {}
    for i, (time_step, mean_r_sq) in enumerate(zip(ml_time_steps, ml_mean_r_squared)):
        key = f"Predicted Phi{int(time_step)}"
        ml_results_for_viz[key] = {
            'avg_radius': np.sqrt(mean_r_sq),  # Convert back to radius
            'actual_grains': 1  # Placeholder to indicate valid data
        }

    pf_results_for_viz = {
        'time_points': pf_results['time_steps'],
        'results': []
    }
    for mean_r_sq in pf_results['mean_r_squared']:
        pf_results_for_viz['results'].append({
            'avg_radius': np.sqrt(mean_r_sq),
            'actual_grains': 1
        })

    # Save joblib files
    ml_joblib_path = os.path.join(output_dir, "ml_kinetics_results.joblib")
    pf_joblib_path = os.path.join(output_dir, "pf_kinetics_results.joblib")

    joblib.dump(ml_results_for_viz, ml_joblib_path)
    joblib.dump(pf_results_for_viz, pf_joblib_path)

    # 8. Generate comparison plot
    print("8. Generating kinetics comparison plot...")
    plot_grain_kinetics_comparison(
        ml_joblib_path=ml_joblib_path,
        pf_joblib_path=pf_joblib_path,
        output_filename=os.path.join(output_dir, 'grain_kinetics_comparison.png')
    )

    # 9. Visualize evolution process
    print("9. Visualizing evolution process...")
    evolution_dir = os.path.join(output_dir, "evolution_visualization")
    key_indices = [0, len(saved_model_data) // 4, len(saved_model_data) // 2,
                   3 * len(saved_model_data) // 4, -1]
    plot_evolution_process(saved_model_data, evolution_dir, key_indices)

    # 10. Compile results
    results = {
        'ml_kinetics': {
            'time_steps': ml_time_steps,
            'mean_r_squared': ml_mean_r_squared,
            'slope': ml_slope,
            'intercept': ml_intercept,
            'r_value': ml_r_value,
            'p_value': ml_p_value,
            'std_err': ml_std_err
        },
        'pf_kinetics': {
            'time_steps': pf_results['time_steps'],
            'mean_r_squared': pf_results['mean_r_squared'],
            'slope': pf_slope,
            'intercept': pf_intercept,
            'r_value': pf_r_value,
            'p_value': pf_p_value,
            'std_err': pf_std_err
        },
        'parameters': {
            'grain_num': grain_num,
            'grid_size': grid_size,
            'max_steps': max_steps,
            'save_interval': save_interval,
            'random_seed': random_seed
        }
    }

    # Save analysis results
    np.savez(
        os.path.join(output_dir, 'kinetics_analysis_data.npz'),
        **{k: v for k, v in results.items() if k != 'parameters'}
    )

    print("=" * 80)
    print("Grain Growth Kinetics Analysis Completed!")
    print(f"ML Model - Slope: {ml_slope:.6f}, R²: {ml_r_value ** 2:.6f}")
    print(f"PF Simulation - Slope: {pf_slope:.6f}, R²: {pf_r_value ** 2:.6f}")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)

    return results


def main():
    """Main function - run kinetics analysis."""

    # Configuration parameters
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'best_checkpoint.pt'
    )
    phase_field_h5_path = os.path.join(
        repo_root,
        'dataset',
        'grain_growth',
        'Test',
        '1024',
        'Isothermal1023_10241024_I1024_J1024_ndt100_mint2000_maxt100000_cleaned75.h5'
    )
    output_dir = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'kinetics_analysis'
    )

    # Analysis parameters
    config = {
        'grain_num': 1024*5,
        'grid_size': 1024,
        'max_steps': 500,
        'save_interval': 5,
        'random_seed': 123,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }

    # Run analysis
    results = analyze_grain_kinetics(
        model_path=model_path,
        phase_field_h5_path=phase_field_h5_path,
        output_dir=output_dir,
        **config
    )

    print("Kinetics analysis completed!")


if __name__ == "__main__":
    main()
