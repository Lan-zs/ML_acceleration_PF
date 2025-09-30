"""
Main experiment runner for grain growth prediction models.

This script orchestrates the complete experimental pipeline including
data loading, model training, evaluation, and analysis.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import random
import numpy as np
import h5py
import json
from typing import Dict, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet
from training.dataloader_temp import create_data_loader
from training.trainer_temp import onestep_training
from evaluation.evaluator_temp import model_testing


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module) -> float:
    """Count model parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def measure_inference_speed(
        model: nn.Module,
        input_size: Tuple[int, int, int, int] = (1, 1, 1024, 1024),
        num_runs: int = 100,
        warmup: int = 50
) -> Tuple[float, float]:
    """
    Measure model inference speed.

    Args:
        model: PyTorch model to measure.
        input_size: Input tensor size (batch_size, channels, height, width).
        num_runs: Number of runs for averaging.
        warmup: Number of warmup runs.

    Returns:
        tuple: (avg_time_ms, throughput_samples_per_sec)
    """
    device = next(model.parameters()).device
    model.eval()

    print(f'Device: {device.type}')
    print(f'Input size: {input_size}')
    print(f'Warmup runs: {warmup}, Test runs: {num_runs}')

    # Create random input data
    dummy_input = torch.randn(input_size).to(device)

    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                # GPU timing
                torch.cuda.synchronize()
                start = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to milliseconds
            else:
                # CPU timing
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Remove outliers
    if len(times) > 10:
        times.sort()
        times = times[len(times) // 10:-len(times) // 10]  # Remove top and bottom 10%

    # Calculate average time and throughput
    avg_time = sum(times) / len(times)
    throughput = 1000 / avg_time * input_size[0]  # Samples per second

    return avg_time, throughput


def run_experiment(
        base_channels: int,
        num_blocks: int,
        kernel_size: int,
        ndt: int,
        save_path: str,
        train_folder: str,
        test_folder: str,
        evaluation_folder: str,
        num_epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        noise_std: float,
        num_workers: int,
        gpu_id: int,
        variant_prefix: str = "augmentation",
        run_training: bool = True,
        run_evaluation: bool = True
) -> Dict:
    """
    Run a single experiment with specified parameters.
    """
    # Create experiment identifier
    model_name = f"ScaledResNet_C{base_channels}_NB{num_blocks}_K{kernel_size}_ndt{ndt}"
    variant = f"{variant_prefix} {model_name}"
    result_dir = os.path.join(save_path, variant)

    print(f"\n{'=' * 80}")
    print(f"Running experiment: {model_name}")
    print(f"Parameters: C={base_channels}, NB={num_blocks}, K={kernel_size}, ndt={ndt}")
    print(f"Result directory: {result_dir}")
    print(f"{'=' * 80}\n")

    # Setup
    setup_seed(1234)
    torch.multiprocessing.freeze_support()

    print("CUDA Available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create directories
    os.makedirs(result_dir, exist_ok=True)

    # Create model
    model = ResidualConcatNet(
        in_channels=2,
        base_channels=base_channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        act_fn=nn.GELU,
        norm_2d=nn.BatchNorm2d,
        min_value=0.7,
        max_value=1.0
    )

    # Load data
    train_loader = create_data_loader(
        train_folder,
        batch_size=train_batch_size,
        ndt=ndt,
        num_workers=num_workers,
        shuffle=True,
        use_iterable=False
    )

    train_eval_loader = create_data_loader(
        train_folder,
        batch_size=eval_batch_size,
        ndt=ndt,
        num_workers=num_workers,
        shuffle=False,
        use_iterable=False
    )

    test_loader = create_data_loader(
        test_folder,
        batch_size=eval_batch_size,
        ndt=ndt,
        num_workers=num_workers,
        shuffle=False,
        use_iterable=False
    )

    test_loaders = [train_eval_loader, test_loader]

    # Setup training
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )

    torch.cuda.set_device(gpu_id)
    model.cuda()

    # Training
    if run_training:
        onestep_training(
            model, criterion, optimizer, scheduler, variant, train_loader, test_loaders,
            result_dir, num_epochs=num_epochs, noise_std=noise_std, save=True
        )

    # Load best model for evaluation
    try:
        torch.cuda.set_device(gpu_id)
        model.cuda()
        checkpoint = torch.load(os.path.join(result_dir, 'best_checkpoint.pt'), map_location=f"cuda:{gpu_id}")
        model.load_state_dict(checkpoint['model_state'])
        print("Loaded best model checkpoint")
    except FileNotFoundError:
        print(f"Warning: best_checkpoint.pt not found in {result_dir}")
        return {}

    # Model evaluation
    if run_evaluation:
        model.eval()
        evaluation_modes = ['onestep', 'rollout']
        assess_metrics = 'MAE'

        for h5_file in os.listdir(evaluation_folder):
            if h5_file.endswith(".h5"):
                h5_file_path = os.path.join(evaluation_folder, h5_file)
                # 物理参数
                Q = 1.82
                kev = 8.625e-5
                Tmax = 1073.0

                with h5py.File(h5_file_path, 'r') as f:
                    phi_data = f['phi_data'][:]
                    temp_data_ = f['temp_data'][:]
                    temp_data = np.exp(-Q / (kev * temp_data_)) / np.exp(-Q / (kev * Tmax))
                    metadata = dict(f['metadata'].attrs)
                    print(np.max(temp_data))
                    print(np.min(temp_data))

                for evaluation_mode in evaluation_modes:
                    output_dir = os.path.join(
                        result_dir,
                        f"evaluation_{assess_metrics}_adjust",
                        f"{os.path.splitext(h5_file)[0]}_{evaluation_mode}"
                    )

                    print(f"Evaluating: {evaluation_mode} mode on {h5_file}")
                    indices = [0]  # Time indices for detailed analysis

                    model_testing(
                        model, phi_data, temp_data, indices, output_dir,
                        evaluation_mode, assess_metrics='MAE', step_interval=100
                    )


    return {
        "model_name": model_name,
        "base_channels": base_channels,
        "num_blocks": num_blocks,
        "kernel_size": kernel_size,
        "ndt": ndt,
        "result_dir": result_dir
    }


def main():
    """Main experiment execution."""

    # =========================================================================
    # EXPERIMENT PARAMETERS - Modify these for your experiments
    # =========================================================================

    # Paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(repo_root, "non-isothermal_grain_growth", "experiment_results")
    train_folder = os.path.join(repo_root, "dataset", "grain_growth_temp", "Training")
    test_folder = os.path.join(repo_root, "dataset", "grain_growth_temp", "Test")
    evaluation_folder = os.path.join(repo_root, "dataset", "grain_growth_temp", "Evaluation")

    # Training parameters
    num_epochs = 150
    train_batch_size = 4
    eval_batch_size = 40
    learning_rate = 1e-3
    weight_decay = 1e-2
    noise_std = 0.0001

    # Other settings
    num_workers = 1
    gpu_id = 0
    variant_prefix = "augmentation"
    run_training = False
    run_evaluation = True

    # Define experiment configurations
    experiments = [
        {"base_channels": 96, "num_blocks": 10, "kernel_size": 3, "ndt": 1},
        # Add more configurations as needed
    ]

    # =========================================================================
    # RUN EXPERIMENTS
    # =========================================================================

    # Run experiments
    results = []
    for exp in experiments:
        result = run_experiment(
            base_channels=exp["base_channels"],
            num_blocks=exp["num_blocks"],
            kernel_size=exp["kernel_size"],
            ndt=exp["ndt"],

            save_path=save_path,
            train_folder=train_folder,
            test_folder=test_folder,
            evaluation_folder=evaluation_folder,

            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            noise_std=noise_std,

            num_workers=num_workers,
            gpu_id=gpu_id,
            variant_prefix=variant_prefix,
            run_training=run_training,
            run_evaluation=run_evaluation
        )
        results.append(result)

    # Save experiment summary
    summary_file = os.path.join(save_path, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nExperiment summary saved to: {summary_file}")
    print("All experiments completed!")


if __name__ == '__main__':
    main()
