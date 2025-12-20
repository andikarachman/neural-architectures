"""
DenseNet-121: Densely Connected Convolutional Network (Didactic Version)

DenseNet was introduced in:
"Densely Connected Convolutional Networks" — Huang et al. (2017)

THE CORE INNOVATION: DENSE CONNECTIVITY
======================================
Instead of connecting layers in a simple chain, DenseNet connects each layer to ALL
previous layers within a DenseBlock.

If we denote the output of layer l as x_l, then:

  x_l = H_l([x_0, x_1, ..., x_{l-1}])

Where:
  - [ ... ] means CONCATENATION along channels (not addition!)
  - H_l is a small transformation (BN -> ReLU -> Conv ...)

RESNET vs DENSENET (Key Difference)
===================================
ResNet:
  y = x + F(x)          (addition)

DenseNet:
  y = concat(x, F(x))   (concatenation)

This changes the behavior dramatically:
  - ResNet keeps channel size stable; info is merged by addition.
  - DenseNet grows channels over time; features are preserved explicitly.

WHY DENSE CONNECTIONS HELP
==========================
1) FEATURE REUSE:
   Early features (edges, textures) remain available to ALL later layers.

2) BETTER GRADIENT FLOW:
   Many short paths exist from loss to early layers, improving optimization.

3) PARAMETER EFFICIENCY:
   Because features are reused, DenseNet can achieve strong accuracy with fewer
   parameters than similarly accurate plain networks.

BUT: WHAT'S THE TRADEOFF?
=========================
DenseNet is often:
  - memory heavy (because concatenation grows channel dimension)
  - bandwidth heavy (lots of feature maps moving around)

DENSENET BUILDING BLOCKS
========================
DenseNet is built from:
  (A) DenseLayer: produces k new channels (growth rate)
  (B) DenseBlock: stacks many DenseLayers; channels grow linearly
  (C) Transition: reduces spatial resolution and compresses channels

GROWTH RATE (k)
===============
Each DenseLayer adds k feature maps (channels).
If input has C channels, after L layers:
  output channels = C + L * k

COMPRESSION (theta)
===================
Transition layers typically compress channels:
  C_out = floor(theta * C_in)
Common setting: theta = 0.5  (DenseNet-121)

DENSENET-121 (ImageNet) STRUCTURE
=================================
Input: (N, 3, 224, 224)

Stem:
  Conv7x7 stride2 -> BN -> ReLU -> MaxPool  => (N, 64, 56, 56)

DenseBlock1: 6 layers, growth k=32
Transition1: compress + AvgPool2 => spatial downsample (56 -> 28)

DenseBlock2: 12 layers
Transition2: (28 -> 14)

DenseBlock3: 24 layers
Transition3: (14 -> 7)

DenseBlock4: 16 layers
Head:
  BN -> ReLU -> GlobalAvgPool -> FC

Total: 121 layers (counting conv layers and FC in the original convention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        """
        Args:
            in_channels (int): Number of input feature channels.
            growth_rate (int): Number of output feature channels to add.
            bn_size (int): Bottleneck size factor. Default is 4.
        """
        super().__init__()
        inter_channels = bn_size * growth_rate # Intermediate channels after bottleneck

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 
                               kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, 
                               kernel_size=3, padding=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W)
        out = self.bn1(x)                   # out: (N, in_channels, H, W)
        out = F.relu(out, inplace=True)     # out: (N, in_channels, H, W)
        out = self.conv1(out)               # out: (N, inter_channels, H, W)
        
        out = self.bn2(out)                 # out: (N, inter_channels, H, W)   
        out = F.relu(out, inplace=True)     # out: (N, inter_channels, H, W)
        out = self.conv2(out)               # out: (N, growth_rate, H, W)

        out = torch.cat([x, out], dim=1)    # out: (N, in_channels + growth_rate, H, W)
        return out
    

class DenseBlock(nn.Module):
    """
        Dense Block consisting of multiple Dense Layers.
        Channel dimension grows by growth_rate for each layer.
    """
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4):
        """
        Args:
            num_layers (int): Number of dense layers in the block.
            in_channels (int): Number of input feature channels.
            growth_rate (int): Growth rate for each dense layer.
            bn_size (int): Bottleneck size factor. Default is 4.
        """
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate, bn_size)
            layers.append(layer)
            channels += growth_rate  # Update channel count after each layer
        
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels  # Total output channels after block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
        

class TransitionLayer(nn.Module):
    """
        Transition Layer to reduce feature map size and number of channels.
           BN -> ReLU -> 1x1 Conv (channel compression) -> 2x2 AvgPool (spatial downsampling)

        This:
        - Reduces number of channels by compression factor (theta)
        - Halves spatial dimensions
    """
    def __init__(self, in_channels: int, theta: float = 0.5):
        """
        Args:
            in_channels: Number of input feature channels.
            theta: Compression factor for channels. Default is 0.5.
        """
        super().__init__()
        
        out_channels = int(theta * in_channels)
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)                # (N, in_channels, H, W)
        out = F.relu(out, inplace=True) # (N, in_channels, H, W)
        out = self.conv(out)            # (N, in_channels * theta, H, W)
        out = self.avg_pool(out)        # (N, out_channels, H/2, W/2)
        return out



class DenseNet(nn.Module):
    """
    DenseNet for ImageNet-style inputs (224x224), parameterized by block sizes.

    DenseNet-121 uses block_config = [6, 12, 24, 16]
    growth_rate = 32
    theta = 0.5
    bn_size = 4
    """
    def __init__(
            self, 
            num_classes: int = 1000,                # Number of output classes (e.g., 1000 for ImageNet)
            growth_rate: int = 32,                  # Growth rate for dense layers (32 is standard from paper)
            block_config: list = [6, 12, 24, 16],   # Number of layers in each dense block (DenseNet-121 configuration)
            init_channels: int = 64,                # Initial number of feature channels
            bn_size: int = 4,                       # Bottleneck size factor (4 is standard from paper)
            theta: float = 0.5                      # Compression factor for transition layers (0.5 is standard from paper)
    ) -> None:
        super().__init__()

        # Initial convolution and pooling (the "stem")
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=7, stride=2, padding=3, bias=False),  # (N, 64, 112, 112)
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (N, 64, 56, 56)
        )

        channels = init_channels

        # Create Dense Blocks and Transition Layers

        # 1st Dense Block + Transition
        self.block1 = DenseBlock(block_config[0], channels, growth_rate, bn_size)
        channels = self.block1.out_channels
        self.trans1 = TransitionLayer(channels, theta)
        channels = self.trans1.out_channels

        # 2nd Dense Block + Transition
        self.block2 = DenseBlock(block_config[1], channels, growth_rate, bn_size)
        channels = self.block2.out_channels
        self.trans2 = TransitionLayer(channels, theta)
        channels = self.trans2.out_channels

        # 3rd Dense Block + Transition
        self.block3 = DenseBlock(block_config[2], channels, growth_rate, bn_size)
        channels = self.block3.out_channels
        self.trans3 = TransitionLayer(channels, theta)
        channels = self.trans3.out_channels

        # 4th Dense Block (no transition after this)
        self.block4 = DenseBlock(block_config[3], channels, growth_rate, bn_size)
        channels = self.block4.out_channels

        # Final batch norm and classification layer
        self.bn_final = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (N, 3, 224, 224)
        out = self.stem(x)               # (N, 64, 56, 56)

        out = self.block1(out)           # (N, C1, 56, 56)
        out = self.trans1(out)           # (N, C1 * theta, 28, 28)

        out = self.block2(out)           # (N, C2, 28, 28)
        out = self.trans2(out)           # (N, C2 * theta, 14, 14)

        out = self.block3(out)           # (N, C3, 14, 14)
        out = self.trans3(out)           # (N, C3 * theta, 7, 7)

        out = self.block4(out)           # (N, C4, 7, 7)

        out = self.bn_final(out)         # (N, C4, 7, 7)
        out = F.relu(out, inplace=True)  # (N, C4, 7, 7)

        out = F.adaptive_avg_pool2d(out, (1, 1))  # (N, C4, 1, 1)
        out = torch.flatten(out, 1)      # (N, C4)
        out = self.fc(out)               # (N, num_classes)

        return out
    

def densenet121(num_classes: int = 1000) -> DenseNet:
    """
    Constructs a DenseNet-121 model.
    
    Args:
        num_classes (int): Number of output classes. Default is 1000 for ImageNet.
    
    Returns:
        DenseNet: DenseNet-121 model instance.
    """
    return DenseNet(
        num_classes=num_classes,
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        init_channels=64,
        bn_size=4,
        theta=0.5
    )
    
if __name__ == "__main__":
    print("=" * 70)
    print("DenseNet-121 (ImageNet-style) — Didactic Sanity Check")
    print("=" * 70)

    model = densenet121(num_classes=1000)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)

    print("\n[1] Input/Output")
    print("    input :", tuple(x.shape))
    print("    output:", tuple(y.shape))

    print("\n[2] Parameter count")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    total params: {total_params:,}")

    print("\n[3] Parameter breakdown (rough)")
    stem_params = sum(p.numel() for p in model.stem.parameters())
    b1 = sum(p.numel() for p in model.block1.parameters())
    t1 = sum(p.numel() for p in model.trans1.parameters())
    b2 = sum(p.numel() for p in model.block2.parameters())
    t2 = sum(p.numel() for p in model.trans2.parameters())
    b3 = sum(p.numel() for p in model.block3.parameters())
    t3 = sum(p.numel() for p in model.trans3.parameters())
    b4 = sum(p.numel() for p in model.block4.parameters())
    head = sum(p.numel() for p in model.bn_final.parameters()) + sum(p.numel() for p in model.fc.parameters())

    print(f"    stem   : {stem_params:,}")
    print(f"    block1 : {b1:,} | trans1: {t1:,}")
    print(f"    block2 : {b2:,} | trans2: {t2:,}")
    print(f"    block3 : {b3:,} | trans3: {t3:,}")
    print(f"    block4 : {b4:,}")
    print(f"    head   : {head:,}")

    print("\n" + "=" * 70)
    print("Key takeaway: DenseNet grows channels via CONCAT, enabling explicit feature reuse.")
    print("=" * 70)


        