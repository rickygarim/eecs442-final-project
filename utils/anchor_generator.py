import torch

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate anchor boxes at different scales and aspect ratios.

    Args:
        base_size (int): Base size of the anchor.
        ratios (list): Aspect ratios for the anchors.
        scales (list): Scaling factors for the anchors.

    Returns:
        torch.Tensor: Generated anchors.
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * (ratio ** 0.5)
            h = base_size * scale / (ratio ** 0.5)
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors)
