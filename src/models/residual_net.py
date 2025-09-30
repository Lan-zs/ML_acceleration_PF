"""
Residual Concatenation Network.

This module contains the implementation of ResidualConcatNet, a neural network
architecture that uses residual connections with channel concatenation for
predicting grain growth evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPad(nn.Module):
    """
    Circular padding layer for periodic boundary conditions.

    Args:
        padding (int): Padding size for each dimension.
    """

    def __init__(self, padding: int):
        super(CircularPad, self).__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply circular padding to input tensor."""
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')


class DoubleConv(nn.Module):
    """
    Double convolution block with circular padding.

    Consists of two consecutive convolution operations, each followed by
    normalization and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel. Default: 3.
        act_fn: Activation function class. Default: nn.ReLU.
        norm_2d: 2D normalization layer class. Default: nn.BatchNorm2d.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            act_fn=nn.ReLU,
            norm_2d=nn.BatchNorm2d
    ):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            CircularPad(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn(),
            CircularPad(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            norm_2d(out_channels),
            act_fn()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution block."""
        return self.conv(x)


class ResidualConcatNet(nn.Module):
    """
    Residual Concatenation Network for grain growth prediction.

    This network uses residual connections with channel concatenation to
    predict grain growth evolution. The output is scaled between min_value
    and max_value using sigmoid activation.

    Args:
        in_channels (int): Number of input channels. Default: 2.
        base_channels (int): Base number of channels. Default: 64.
        num_blocks (int): Number of residual blocks. Default: 4.
        kernel_size (int): Convolution kernel size. Default: 3.
        act_fn: Activation function class. Default: nn.ReLU.
        norm_2d: 2D normalization layer class. Default: nn.BatchNorm2d.
        min_value (float): Minimum output value for scaling. Default: 0.7.
        max_value (float): Maximum output value for scaling. Default: 1.0.
    """

    def __init__(
            self,
            in_channels: int = 2,
            base_channels: int = 64,
            num_blocks: int = 4,
            kernel_size: int = 3,
            act_fn=nn.ReLU,
            norm_2d=nn.BatchNorm2d,
            min_value: float = 0.7,
            max_value: float = 1.0
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.min_value = min_value
        self.max_value = max_value

        # First convolution layer
        self.first_conv = nn.Sequential(
            CircularPad(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        # Residual blocks with concatenation
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # Each residual block contains:
            # 1. Main path processing module
            # 2. Fusion convolution for concatenated features
            self.res_blocks.append(nn.ModuleList([
                DoubleConv(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # Output layer
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple: (predicted_phi_scaled, delta_phi)
                - predicted_phi_scaled: Scaled prediction output
                - delta_phi: Raw prediction before scaling
        """
        # Initial transformation
        x = self.first_conv(x)

        # Process through residual blocks
        for main_path, fusion_conv in self.res_blocks:
            identity = x
            x = main_path(x)
            # Concatenate along channel dimension
            x = torch.cat([x, identity], dim=1)
            # Reduce channels back to base_channels
            x = fusion_conv(x)

        # Generate output
        delta_phi = self.final_conv(x)
        predicted_phi = delta_phi.squeeze()

        # Apply sigmoid normalization and scale to [min_value, max_value]
        normalized_phi = torch.sigmoid(predicted_phi)
        predicted_phi_scaled = self.min_value + (self.max_value - self.min_value) * normalized_phi

        # Handle batch dimension properly
        if predicted_phi_scaled.dim() == 2:  # Single sample case
            predicted_phi_scaled = predicted_phi_scaled.unsqueeze(0)

        return predicted_phi_scaled.unsqueeze(1), delta_phi


class ResidualConcatNet_Res(nn.Module):
    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_blocks=4,
                 kernel_size=3,
                 act_fn=nn.ReLU,
                 norm_2d=nn.BatchNorm2d,
                 ):
        super().__init__()
        self.num_blocks = num_blocks

        # 第一个卷积层
        self.first_conv = nn.Sequential(
            CircularPad(kernel_size // 2),
            nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, padding=0),
            norm_2d(base_channels),
            act_fn()
        )

        # 残差块 - 通道拼接方式
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # 每个残差块包含一个处理主路径的模块，和一个处理拼接后的模块
            self.res_blocks.append(nn.ModuleList([
                # 主路径卷积，输出通道数与输入相同
                DoubleConv(base_channels, base_channels, kernel_size, act_fn, norm_2d),
                # 拼接后处理的卷积，输入通道数是base_channels*2，输出通道数是base_channels
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
            ]))

        # 输出层
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        current_phi=x
        # 初始转换
        x = self.first_conv(x)

        # 残差块处理
        for main_path, fusion_conv in self.res_blocks:
            identity = x
            x = main_path(x)
            # 拼接残差连接
            x = torch.cat([x, identity], dim=1)  # 通道维度拼接
            x = fusion_conv(x)  # 将拼接后的通道数减少回base_channels

        # 输出处理
        delta_phi = self.final_conv(x)

        # 输出处理
        predicted_phi = delta_phi + current_phi

        return predicted_phi, delta_phi
        # 有bug，当一张图片时有bug，batch为1，上面squeeze直接变成没有维度了，再.unsqueeze(1)会变成(256,1,256)


if __name__ == "__main__":
    """Test the ResidualConcatNet model with different configurations."""
    test_configs = [
        {'in_channels': 1, 'sizes': [(2, 1, 48, 48), (2, 1, 100, 100), (2, 1, 256, 256)]},
    ]

    for config in test_configs:
        print(f"\nTesting model with {config['in_channels']} input channels:")

        # model = ResidualConcatNet(
        model = ResidualConcatNet_Res(
            in_channels=config['in_channels'],
            base_channels=64,
            num_blocks=4,
            kernel_size=3,
            act_fn=nn.ReLU,
            norm_2d=nn.BatchNorm2d
        )

        for size in config['sizes']:
            x = torch.randn(size)
            predicted_phi_scaled, delta_phi = model(x)
            print(f"Input size: {size}")
            print(f"Predicted phi scaled size: {predicted_phi_scaled.shape}")
            print(f"Delta phi size: {delta_phi.shape}")