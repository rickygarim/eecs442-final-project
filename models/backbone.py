import torch
import torchvision
import torch.nn as nn

def build_backbone():
    """
    Returns a backbone network for feature extraction.
    """
    # Load a pre-trained ResNet-50 model
    resnet = torchvision.models.resnet50(pretrained=True)
    
    # Remove the fully connected layers and pool
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    
    # Specify the number of output channels for the backbone
    backbone.out_channels = 2048  # ResNet-50's output channels
    return backbone
