"""
Receptive field statistics runner for non-isothermal grain growth models.

Loads a trained model, configures dataset/evaluation paths relative to the repo,
and runs receptive field analysis and summary generation.
"""

import torch
import torch.nn as nn
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.residual_net import ResidualConcatNet
from analysis.receptive_field_analysis_temp import analyze_model_receptive_field


def load_model(checkpoint_path):
    """Load model from .pt checkpoint path."""
    # Determine model type
    if 'resnet' in checkpoint_path.lower():
        print("Loading ResNet model")
        model = ResidualConcatNet(
            in_channels=2,
            base_channels=96,
            num_blocks=10,
            kernel_size=3,
            act_fn=nn.GELU,  # GELU activation function
            norm_2d=nn.BatchNorm2d,
            min_value=0.7,  # lower scaling bound
            max_value=1.0  # upper scaling bound
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    return model


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
    result_dir = os.path.join(
        repo_root,
        'non-isothermal_grain_growth',
        'experiment_results',
        'augmentation ScaledResNet_C96_NB10_K3_ndt1'
    )

    # 加载模型
    print("正在加载模型...")
    model = load_model(checkpoint_path)

    # Analyze receptive fields
    analyze_model_receptive_field(
        model,
        model_evaluation_H5_folder,
        result_dir,
        test_image_indices=[0],
        num_points_per_category=10
    )



if __name__ == "__main__":
    main()