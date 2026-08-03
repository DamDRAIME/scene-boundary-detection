import torch

from sbd.coregistration.distance import ProximityMetric, compute_pairwise_similarity


def extract_anchors(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: ProximityMetric,
    similarity_threshold: float,
    min_count: int = 1,
    max_count: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairwise_distance_matrix = compute_pairwise_similarity(a, b, metric)
    above_threshold = pairwise_distance_matrix > similarity_threshold
    count_gt_threshold_per_row = above_threshold.count_nonzero(dim=1)
    row_mask = (min_count <= count_gt_threshold_per_row) & (count_gt_threshold_per_row < max_count)
    count_gt_threshold_per_col = above_threshold.count_nonzero(dim=0)
    col_mask = (min_count <= count_gt_threshold_per_col) & (count_gt_threshold_per_col < max_count)
    anchors = above_threshold & row_mask.unsqueeze(1) & col_mask.unsqueeze(0)
    return anchors.count_nonzero(dim=1) > 0, anchors.count_nonzero(dim=0) > 0
