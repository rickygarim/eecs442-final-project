from torchvision.ops import roi_pool

def apply_roi_pooling(feature_map, proposals, output_size=(7, 7), spatial_scale=1.0):
    """
    Apply ROI pooling on a feature map using proposals.

    Args:
        feature_map (torch.Tensor): Feature map from the backbone.
        proposals (torch.Tensor): Proposals (regions of interest).
        output_size (tuple): Output size of each pooled region.
        spatial_scale (float): Scaling factor to map proposals to feature map.

    Returns:
        torch.Tensor: Pooled feature regions.
    """
    pooled_regions = roi_pool(
        feature_map,
        proposals,
        output_size=output_size,
        spatial_scale=spatial_scale
    )
    return pooled_regions
