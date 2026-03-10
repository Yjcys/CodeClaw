import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, upscale_factor=4):
        super(Generator, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Sub-pixel convolutional layer for upsampling
        self.sub_pixel = nn.Conv2d(64, 64 * (upscale_factor**2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(16)]
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Sub-pixel convolution for upsampling
        x = self.sub_pixel(x)
        x = self.pixel_shuffle(x)
        
        x = self.res_blocks(x)
        
        x = self.conv2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = self.relu(x)
        return x