import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from .backbone import build_backbone

def get_faster_rcnn_model(num_classes):
    """
    Returns a Faster R-CNN model with a custom backbone.
    """
    # Load a backbone network
    backbone = build_backbone()

    # Anchor generator for RPN
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # ROI Pooling layer
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],  # Single feature map
        output_size=7,
        sampling_ratio=2
    )

    # Build the Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model
