import torch
import torch.nn as nn

from architecture.attention import MultiHeadAttention

class PreActivationResidualBlock(nn.Module):
    """
    Residual block with pre activation.
    
    Args:
     - in_channels (int): Number of input channels.
     - out_channels (int): Number of output channels.
     - kernel_size (int): Size of the convolutional kernel.
     - stride (int): Stride of the convolution.
     - padding (int): Padding for the convolution.

    Note: 
     - If in_channels != out_channels, a 1x1 convolution is used to match the dimensions.

    Refernce:
     - https://github.com/hkproj/pytorch-stable-diffusion
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 32) -> None:
        super(PreActivationResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        if in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pre activated residual block.
        
        Args:
         - x (torch.Tensor): Input tensor of shape (batch_Size, in_channels, height, width).
            
        Returns:
         - torch.Tensor: Output tensor of shape (batch_Size, out_channels, height, width) after passing through the residual block.
        """
        res = self.res(x)
        x = self.norm1(x)
        x = nn.SiLU()(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.SiLU()(x)
        x = self.conv2(x)
        return x + res
    
class AttentionBlock(nn.Module):
    """
    Attention block with group normalization and self-attention.

    Args:
     - channels (int): Number of input channels.
     - groups (int): Number of groups for group normalization.
    
    Refernce:
     - https://github.com/hkproj/pytorch-stable-diffusion
    """
    def __init__(self, channels: int, groups: int = 32) -> None:
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attention = SelfAttention(1, channels)
        self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention block.

        Args:
         - x (torch.Tensor): Input tensor of shape (batch_Size, features, height, width).
        
        Returns:
         - torch.Tensor: Output tensor of shape (batch_Size, features, height, width) after passing through the attention block.

        Note:
         - Need to reshape the input tensor to perform self-attention. Specificallty, sequence length is "height * width" and "features" defines the dimension of each word's embedding.
        """
        res = self.res(x) 
        x = self.norm(x)

        n, c, h, w = x.shape
        
        # (batch_Size, features, height, width) -> (batch_Size, features, height * width)
        x = x.view((n, c, h * w))
        
        # (batch_Size, features, height * width) -> (batch_Size, height * width, features). 
        # Each pixel becomes a feature of size "features", the sequence length is "height * width".
        x = x.transpose(-1, -2)
        
        # Perform self-attention without causal mask
        x = self.attention(x, causal_mask=False)
        
        # (batch_Size, height * width, features) -> (batch_Size, features, height * width)
        x = x.transpose(-1, -2)
        
        # (batch_Size, features, height * width) -> (batch_Size, features, height, width)
        x = x.view((n, c, h, w))
        
        return x + res
        
        
    