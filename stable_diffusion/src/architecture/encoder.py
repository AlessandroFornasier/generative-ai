import os
import sys

import torch
import torch.nn as nn

from pathlib import Path
from typing import List, OrderedDict

from architecture.blocks import PreActivationResidualBlock, AttentionBlock

parent_path = Path(__file__).resolve().parent.parent.parent.parent
print(parent_path)

autoencoders_path = parent_path / 'autoencoders/src'
stable_difusion_path = parent_path / 'stable_diffusion/src'
sys.path.append(str(autoencoders_path))
sys.path.append(str(stable_difusion_path))

class SDVAE():
    def encoder(self) -> OrderedDict[str, nn.Module]:
      """
      Defines the VAE encoder for stable diffusion.
      
      Note:
        padding=1 means the width and height will increase by 2
        Out_Height = In_Height + Padding_Top + Padding_Bottom (which means +2)
        Out_Width = In_Width + Padding_Left + Padding_Right (which means +2)
        This will compensate for the Kernel size of 3

       Refernce:
         - https://github.com/hkproj/pytorch-stable-diffusion
      """
      encoder = [
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        PreActivationResidualBlock(128, 128),
        PreActivationResidualBlock(128, 128),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        PreActivationResidualBlock(128, 256), 
        PreActivationResidualBlock(256, 256), 
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
        PreActivationResidualBlock(256, 512), 
        PreActivationResidualBlock(512, 512), 
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
        PreActivationResidualBlock(512, 512), 
        PreActivationResidualBlock(512, 512), 
        PreActivationResidualBlock(512, 512), 
        AttentionBlock(512), 
        PreActivationResidualBlock(512, 512), 
        nn.GroupNorm(32, 512), 
        nn.SiLU(),             
        nn.Conv2d(512, 8, kernel_size=3, padding=1), 
        nn.Conv2d(8, 8, kernel_size=1, padding=0), 
      ]
      return OrderedDict(encoder)

    def decoder(self) -> OrderedDict[str, nn.Module]:
      """
      Defines the VAE decoder for stable diffusion.
      """
      pass

