
from typing import Union

import torch
import torch.nn as nn


cfgs: dict[str, list[Union[str, int]]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: list[Union[str, int]], in_channels: int = 3) -> nn.Sequential:
    """
    Build the VGG feature extractor:
        Conv3x3 -> ReLU -> MaxPool (repeated according to cfg)

    With 224x224 input and 5 pools:
        224 -> 112 -> 56 -> 28 -> 14 -> 7
    
    So the final feature map size is 7x7 with 512 channels.
    """
    layers = []
    c_in = in_channels
    
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            c_out = int(v)
            conv2d = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            c_in = c_out
    
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, cfg_name: str, num_classes: int = 1000, in_channels: int = 3, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = make_layers(cfgs[cfg_name], in_channels=in_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),               # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),                      # (N, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),                # (N, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        
        x = torch.flatten(x, 1)     # Flatten from (N, 512, 7, 7) to (N, 512*7*7)
        x = self.classifier(x)
        return x

def vgg11(num_classes=1000, in_channels=3):
    return VGG("VGG11", num_classes=num_classes, in_channels=in_channels)


def vgg13(num_classes=1000, in_channels=3):
    return VGG("VGG13", num_classes=num_classes, in_channels=in_channels)

def vgg16(num_classes=1000, in_channels=3):
    return VGG("VGG16", num_classes=num_classes, in_channels=in_channels)

def vgg19(num_classes=1000, in_channels=3):
    return VGG("VGG19", num_classes=num_classes, in_channels=in_channels)

if __name__ == "__main__":
    model = vgg16(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)  # torch.Size([1, 1000])