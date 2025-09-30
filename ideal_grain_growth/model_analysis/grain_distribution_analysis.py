"""
Grain Size Distribution Analysis

This module analyzes steady-state grain size distributions by comparing
deep learning model predictions with phase field simulations.
"""

import os
import sys
import numpy as np
import h5py
import joblib
from glob import glob
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.grain_growth_quantitative_validation import (
    extract_ml_grain_distributions,
    extract_pf_grain_distributions,
    calculate_average_steady_state_distribution,
    save_combined_steady_state_data
)
from utils.visualization import (
    plot_steady_state_distribution_comparison,
    plot_distribution_evolution_summary
)


def check_and_extract_distributions(
        ml_dat_dir: str,
        pf_h5_path: str,
        ml_output_dir: str,
        pf_output_dir: str,
        force_regenerate: bool = False,
        pf_analysis_interval: int = 5
) -> Tuple[str, str]:
    """
    Check and extract distribution data for both ML and PF simulations.

    Args:
        ml_dat_dir: Directory containing ML .dat files.
        pf_h5_path: Path to PF h5 file.
        ml_output_dir: Output directory for ML distributions.
        pf_output_dir: Output directory for PF distributions.
        force_regenerate: Whether to force regeneration.
        pf_analysis_interval: Interval for PF analysis.

    Returns:
        ml_output_dir: ML distribution directory.
        pf_output_dir: PF distribution directory.
    """
    # Check ML distributions
    if os.path.exists(ml_output_dir) and not force_regenerate:
        ml_files = glob(os.path.join(ml_output_dir, "*_distribution.joblib"))
        if ml_files:
            print(f"ML distribution data already exists: {ml_output_dir}")
        else:
            force_regenerate = True
    else:
        force_regenerate = True

    if force_regenerate or not os.path.exists(ml_output_dir):
        print("Extracting ML grain size distributions...")
        dat_files = sorted(glob(os.path.join(ml_dat_dir, "*.dat")))
        if not dat_files:
            raise ValueError(f"No .dat files found in {ml_dat_dir}")
        extract_ml_grain_distributions(dat_files, ml_output_dir)

    # Check PF distributions
    pf_regenerate = force_regenerate
    if os.path.exists(pf_output_dir) and not force_regenerate:
        pf_files = glob(os.path.join(pf_output_dir, "*_distribution.joblib"))
        if pf_files:
            print(f"PF distribution data already exists: {pf_output_dir}")
        else:
            pf_regenerate = True
    else:
        pf_regenerate = True

    if pf_regenerate or not os.path.exists(pf_output_dir):
        print("Extracting PF grain size distributions...")
        extract_pf_grain_distributions(pf_h5_path, pf_output_dir, pf_analysis_interval)

    return ml_output_dir, pf_output_dir


def analyze_grain_distributions(
        model_rollout_data_dir: str,
        phase_field_h5_path: str,
        output_dir: str,
        ml_steady_range: Tuple[int, int] = (400, 500),
        pf_steady_range: Tuple[int, int] = (800, 1000),
        pf_analysis_interval: int = 5,
        force_regenerate: bool = False
) -> Dict:
    """
    Analyze grain size distributions comparing ML model with phase field simulation.

    Args:
        model_rollout_data_dir: Directory containing model rollout .dat files.
        phase_field_h5_path: Path to phase field simulation data.
        output_dir: Output directory.
        ml_steady_range: Time range for ML steady state analysis.
        pf_steady_range: Time range for PF steady state analysis.
        pf_analysis_interval: Interval for PF analysis.
        force_regenerate: Whether to force regeneration of distribution data.

    Returns:
        results: Analysis results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Starting Grain Size Distribution Analysis")
    print("=" * 80)

    # 1. Set up separate directories for ML and PF distributions
    ml_distribution_dir = os.path.join(output_dir, "ml_distributions")
    pf_distribution_dir = os.path.join(output_dir, "pf_distributions")

    # 2. Check and extract distribution data
    print("1. Checking and extracting distribution data...")
    ml_dist_dir, pf_dist_dir = check_and_extract_distributions(
        ml_dat_dir=model_rollout_data_dir,
        pf_h5_path=phase_field_h5_path,
        ml_output_dir=ml_distribution_dir,
        pf_output_dir=pf_distribution_dir,
        force_regenerate=force_regenerate,
        pf_analysis_interval=pf_analysis_interval
    )

    # 3. Create distribution evolution summary
    print("2. Creating distribution evolution summary...")
    evolution_output_dir = os.path.join(output_dir, "distribution_evolution")
    plot_distribution_evolution_summary(
        ml_distribution_dir=ml_dist_dir,
        pf_distribution_dir=pf_dist_dir,
        output_dir=evolution_output_dir
    )

    # 4. Calculate steady-state distributions
    print("3. Calculating steady-state distributions...")

    try:
        # ML steady state
        ml_bin_centers, ml_avg_distribution = calculate_average_steady_state_distribution(
            distribution_dir=ml_dist_dir,
            time_range=ml_steady_range,
            data_type='ml'
        )
        print(f"ML steady state calculated for range {ml_steady_range}")

        # PF steady state
        pf_bin_centers, pf_avg_distribution = calculate_average_steady_state_distribution(
            distribution_dir=pf_dist_dir,
            time_range=pf_steady_range,
            data_type='pf'
        )
        print(f"PF steady state calculated for range {pf_steady_range}")

    except Exception as e:
        print(f"Error calculating steady-state distributions: {e}")
        return {}

    # 5. Save combined steady-state data
    print("4. Saving combined steady-state data...")
    combined_data_file = os.path.join(output_dir, "combined_steady_state.joblib")
    save_combined_steady_state_data(
        ml_bin_centers=ml_bin_centers,
        ml_avg_distribution=ml_avg_distribution,
        pf_bin_centers=pf_bin_centers,
        pf_avg_distribution=pf_avg_distribution,
        ml_time_range=ml_steady_range,
        pf_time_range=pf_steady_range,
        output_file=combined_data_file
    )

    # 6. Create steady-state comparison plot
    print("5. Creating steady-state comparison plot...")
    comparison_plot_file = os.path.join(output_dir, 'steady_state_distribution_comparison.png')

    # 调用修改后的可视化函数
    ml_bin_centers_viz, ml_distribution_viz, _ = plot_steady_state_distribution_comparison(
        combined_data_file=combined_data_file,
        output_filename=comparison_plot_file
    )

    # 7. Statistical analysis
    print("6. Performing statistical analysis...")

    def calculate_distribution_moments(bin_centers, distribution):
        """Calculate statistical moments of distribution."""
        # Normalize distribution
        norm_dist = distribution / np.trapz(distribution, bin_centers)

        # First moment (mean)
        mean = np.trapz(bin_centers * norm_dist, bin_centers)

        # Second moment (variance)
        variance = np.trapz((bin_centers - mean)**2 * norm_dist, bin_centers)

        # Skewness and kurtosis
        skewness = np.trapz((bin_centers - mean)**3 * norm_dist, bin_centers) / (variance**1.5)
        kurtosis = np.trapz((bin_centers - mean)**4 * norm_dist, bin_centers) / (variance**2)

        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    ml_moments = calculate_distribution_moments(ml_bin_centers, ml_avg_distribution)
    pf_moments = calculate_distribution_moments(pf_bin_centers, pf_avg_distribution)

    # 8. Compile results
    results = {
        'ml_steady_state': {
            'time_range': ml_steady_range,
            'bin_centers': ml_bin_centers,
            'distribution': ml_avg_distribution,
            'moments': ml_moments,
            'distribution_dir': ml_dist_dir
        },
        'pf_steady_state': {
            'time_range': pf_steady_range,
            'bin_centers': pf_bin_centers,
            'distribution': pf_avg_distribution,
            'moments': pf_moments,
            'distribution_dir': pf_dist_dir
        },
        'parameters': {
            'pf_analysis_interval': pf_analysis_interval,
            'force_regenerate': force_regenerate
        }
    }

    # 9. Save analysis results
    analysis_data_file = os.path.join(output_dir, 'distribution_analysis_results.npz')
    np.savez(
        analysis_data_file,
        ml_bin_centers=ml_bin_centers,
        ml_distribution=ml_avg_distribution,
        pf_bin_centers=pf_bin_centers,
        pf_distribution=pf_avg_distribution,
        ml_moments=np.array([ml_moments[k] for k in ['mean', 'std', 'skewness', 'kurtosis']]),
        pf_moments=np.array([pf_moments[k] for k in ['mean', 'std', 'skewness', 'kurtosis']])
    )

    # 10. Save detailed results
    detailed_results_file = os.path.join(output_dir, 'detailed_analysis_results.joblib')
    joblib.dump(results, detailed_results_file)

    print("=" * 80)
    print("Grain Size Distribution Analysis Completed!")
    print(f"ML Distribution - Mean: {ml_moments['mean']:.3f}, Std: {ml_moments['std']:.3f}")
    print(f"PF Distribution - Mean: {pf_moments['mean']:.3f}, Std: {pf_moments['std']:.3f}")
    print(f"ML steady state range: {ml_steady_range}")
    print(f"PF steady state range: {pf_steady_range}")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)

    return results


def main():
    """Main function - run distribution analysis."""

    # Configuration parameters
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_rollout_data_dir = os.path.join(
        repo_root,
        'ideal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C32_NB8_K3_ndt1',
        'kinetics_analysis',
        'model_rollout_data'
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
        'distribution_analysis'
    )


    # Analysis parameters
    config = {
        # 'ml_steady_range': (170, 200),  # Adjust based on your analysis
        # 'pf_steady_range': (170, 200),  # Adjust based on your analysis
        'ml_steady_range': (60, 130),  # Adjust based on your analysis
        'pf_steady_range': (60, 130),  # Adjust based on your analysis
        'pf_analysis_interval': 5,  # Analyze every 5 frames
        'force_regenerate': False
    }

    # Run analysis
    results = analyze_grain_distributions(
        model_rollout_data_dir=model_rollout_data_dir,
        phase_field_h5_path=phase_field_h5_path,
        output_dir=output_dir,
        **config
    )

    print("Distribution analysis completed!")

    # Print summary statistics
    if results:
        ml_stats = results['ml_steady_state']['moments']
        pf_stats = results['pf_steady_state']['moments']

        print("\nSummary Statistics:")
        print(f"ML - Mean: {ml_stats['mean']:.3f}, Std: {ml_stats['std']:.3f}")
        print(f"PF - Mean: {pf_stats['mean']:.3f}, Std: {pf_stats['std']:.3f}")
        print(f"Difference in Mean: {abs(ml_stats['mean'] - pf_stats['mean']):.3f}")


if __name__ == "__main__":
    main()
