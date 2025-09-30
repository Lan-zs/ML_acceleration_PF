"""
One-step training module for grain growth prediction models.

This module provides functionality for training neural networks on
phase field evolution prediction tasks using one-step prediction.
"""

import os
import time
import joblib
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader

# from ..utils.visualization import model_performance_during_training,plot_train_loss_curve

# 将 src 的父目录（即项目根目录）添加到 sys.path
# __file__ 是当前文件 trainer.py 的路径
# os.path.dirname() 用于获取目录名
# 我们需要向上追溯三层：trainer.py -> training -> src -> 项目根目录

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 然后，将原来的相对导入改成从 src 开始的绝对导入
from src.utils.visualization import model_performance_during_training, plot_train_loss_curve


def augment_data(
        current_phi: np.ndarray,
        next_phi: np.ndarray,
        temp_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply data augmentation to phase field data.

    Randomly applies one of 8 transformations: 4 rotations and 4 mirror operations.

    Args:
        current_phi: Current time step phase field data.
        next_phi: Next time step phase field data.
        temp_data: Temperature field data.

    Returns:
        tuple: Augmented (current_phi, next_phi, temp_data).
    """
    # Define 8 augmentation modes: rotations and reflections
    augmentations = [
        lambda x: x,  # Original
        lambda x: np.rot90(x, k=1, axes=(1, 2)),  # Rotate 90°
        lambda x: np.rot90(x, k=2, axes=(1, 2)),  # Rotate 180°
        lambda x: np.rot90(x, k=3, axes=(1, 2)),  # Rotate 270°
        lambda x: np.flip(x, axis=1),  # Vertical flip
        lambda x: np.flip(x, axis=2),  # Horizontal flip
        lambda x: np.flip(np.rot90(x, k=1, axes=(1, 2)), axis=1),  # Rotate 90° + vertical flip
        lambda x: np.flip(np.rot90(x, k=1, axes=(1, 2)), axis=2),  # Rotate 90° + horizontal flip
    ]

    # Randomly select augmentation mode
    mode = np.random.choice(len(augmentations))
    augmentation = augmentations[mode]

    # Apply augmentation
    current_phi_aug = augmentation(current_phi)
    next_phi_aug = augmentation(next_phi)
    temp_data_aug = augmentation(temp_data)

    return current_phi_aug, next_phi_aug, temp_data_aug


def onestep_training(
        model: nn.Module,
        criterion: torch.nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        variant: str,
        train_loader: DataLoader,
        test_loaders: List[DataLoader],
        result_dir: str,
        num_epochs: int = 100,
        noise_std: float = 0.0001,
        save: bool = True
) -> None:
    """
    Train a model for one-step phase field prediction.

    Args:
        model: Neural network model to train.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        variant: Training variant identifier.
        train_loader: Training data loader.
        test_loaders: List of test data loaders [train_eval, test].
        result_dir: Directory to save results.
        num_epochs: Number of training epochs.
        noise_std: Standard deviation for noise augmentation.
        save: Whether to save model and results.
    """
    start_time = time.time()

    # Setup directories
    if save:
        os.makedirs(result_dir, exist_ok=True)
        mpdt_dir = os.path.join(result_dir, "model_performance_during_training")
        os.makedirs(mpdt_dir, exist_ok=True)


    # Training tracking
    train_losses = []
    test_losses = []
    best_losses = []
    optimizer_lrs = []
    best_loss = float('inf')

    print(f"Starting training for {variant}")
    print(f"Training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Training phase
        for current_phi, next_phi, temp_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move data to GPU
            current_phi = current_phi.cuda()
            next_phi = next_phi.cuda()
            temp_data = temp_data.cuda()

            # Apply data augmentation if specified
            if "augmentation" in variant:
                current_phi_np = current_phi.cpu().numpy()
                next_phi_np = next_phi.cpu().numpy()
                temp_data_np = temp_data.cpu().numpy()

                current_phi, next_phi, temp_data = augment_data(
                    current_phi_np, next_phi_np, temp_data_np
                )

                current_phi = torch.tensor(current_phi.copy(), dtype=torch.float32).cuda()
                next_phi = torch.tensor(next_phi.copy(), dtype=torch.float32).cuda()
                temp_data = torch.tensor(temp_data.copy(), dtype=torch.float32).cuda()

            # Add noise if specified
            if "noise" in variant:
                noise = torch.randn_like(current_phi) * noise_std
                current_phi = current_phi + noise

            # Prepare input (only phase field for this model)
            inputs = current_phi.unsqueeze(1)

            # Forward pass
            predicted_phi_scaled, delta_phi = model(inputs)
            predicted_phi_scaled = predicted_phi_scaled.squeeze()

            # Calculate loss
            loss = criterion(predicted_phi_scaled, next_phi.squeeze())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Record learning rate
        optimizer_lr = optimizer.param_groups[0]['lr']
        optimizer_lrs.append(optimizer_lr)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for i, test_loader in enumerate(test_loaders):
                plot_visualization = True
                test_loss = 0
                num_test_batches = 0

                for current_phi, next_phi, temp_data in test_loader:
                    # Move data to GPU
                    current_phi = current_phi.cuda()
                    next_phi = next_phi.cuda()
                    temp_data = temp_data.cuda()

                    # Prepare input
                    inputs = current_phi.unsqueeze(1)

                    # Forward pass
                    predicted_phi_scaled, delta_phi = model(inputs)
                    predicted_phi_scaled = predicted_phi_scaled.squeeze()

                    # Calculate test loss
                    loss = criterion(predicted_phi_scaled, next_phi.squeeze())
                    test_loss += loss.item()
                    num_test_batches += 1

                    # Visualization during training
                    if epoch % 5 == 0 and save and plot_visualization:
                        dataset_name = 'Train' if i == 0 else f'Test{i}'
                        model_performance_during_training(
                            current_phi, delta_phi, predicted_phi_scaled,
                            next_phi, epoch, loss.item(), dataset_name, mpdt_dir
                        )
                        plot_visualization = False

                # Record losses
                avg_test_loss = test_loss / num_test_batches if num_test_batches > 0 else 0

                if i == 0:
                    train_losses.append(avg_test_loss)
                elif i == 1:
                    test_losses.append(avg_test_loss)
                    # Update learning rate scheduler
                    scheduler.step(avg_test_loss)

        # Save best model
        if test_losses and test_losses[-1] < best_loss and save:
            best_loss = test_losses[-1]
            best_checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_loss
            }
            torch.save(best_checkpoint, os.path.join(result_dir, 'best_checkpoint.pt'))
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_model.pt'))

        best_losses.append(best_loss)

        # Save training progress
        if save:
            joblib.dump(
                (train_losses, test_losses, best_losses, optimizer_lrs),
                os.path.join(result_dir, "loss_curve.pkl")
            )
            plot_train_loss_curve(
                train_losses, test_losses, best_losses, optimizer_lrs, result_dir, variant
            )

        # Print progress
        train_loss_str = f"{train_losses[-1]:.6e}" if train_losses else "N/A"
        test_loss_str = f"{test_losses[-1]:.6e}" if test_losses else "N/A"

        print(f"{variant}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss_str}, "
              f"Test Loss: {test_loss_str}, "
              f"Best Loss: {best_loss:.6e}")
        print(f"Learning Rate: {optimizer_lr:.6e}")

    # Save final model
    if save:
        torch.save(model.state_dict(), os.path.join(result_dir, 'final_model.pt'))
        joblib.dump(
            (train_losses, test_losses, best_losses, optimizer_lrs),
            os.path.join(result_dir, "loss_curve.pkl")
        )

    end_time = time.time()
    training_time = int(end_time - start_time)

    if save:
        with open(os.path.join(result_dir, f"training_time_{training_time}_s.txt"), 'w') as f:
            f.write(f"Training completed in {training_time} seconds\n")

    print('Training completed successfully!')

if __name__ == '__main__':
    pass