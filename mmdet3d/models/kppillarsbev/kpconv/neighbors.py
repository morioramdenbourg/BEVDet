""" Implementation from https://github.com/qinzheng93/Easy-KPConv/blob/master/setup.py. """
import torch

from typing import Optional, Tuple
from pykeops.torch import LazyTensor

def find_neighbors(query, support, query_length, support_length, search_radius, neighbor_limit):
    device = query.device
    query_lengths = torch.Tensor([query_length]).int().to(device)
    support_lengths = torch.Tensor([support_length]).int().to(device)
    return radius_search_pack_mode(
        query, support, query_lengths, support_lengths, search_radius, neighbor_limit
    )

def keops_knn(q_points: torch.Tensor, s_points: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    return knn_distances, knn_indices

def _get_indices_from_lengths(lengths: torch.Tensor, num_items: int) -> torch.Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * num_items, i * num_items + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    return indices

def batch_to_pack(batch_tensor: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Tensor from batch mode to stack mode with masks.

    Args:
        batch_tensor (Tensor): the input tensor in batch mode (B, N, C) or (B, N).
        masks (BoolTensor): the masks of items of each sample in the batch (B, N).

    Returns:
        A Tensor in pack mode in the shape of (M, C) or (M).
        A LongTensor of the length of each sample in the batch in the shape of (B).
    """
    if masks is not None:
        pack_tensor = batch_tensor[masks]
        lengths = masks.sum(dim=1)
    else:
        lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long).cuda()
        pack_tensor = batch_tensor
    return pack_tensor, lengths

def pack_to_batch(pack_tensor: torch.Tensor, lengths: torch.Tensor, max_length=None, fill_value=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert Tensor from pack mode to batch mode.

    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (LongTensor): The number of items of each sample in the batch (B)
        max_length (int, optional): The maximal length of each sample in the batch.
        fill_value (float or int or bool): The default value in the empty regions. Default: 0.

    Returns:
        A Tensor in stack mode in the shape of (B, N, C), where N is max(lengths).
        A BoolTensor of the masks of each sample in the batch in the shape of (B, N).
    """
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = lengths.max().item()
    tgt_indices = _get_indices_from_lengths(lengths, max_length)

    num_channels = pack_tensor.shape[1]
    batch_tensor = pack_tensor.new_full(size=(batch_size * max_length, num_channels), fill_value=fill_value)
    batch_tensor[tgt_indices] = pack_tensor
    batch_tensor = batch_tensor.view(batch_size, max_length, num_channels)

    masks = torch.zeros(size=(batch_size * max_length,), dtype=torch.bool).cuda()
    masks[tgt_indices] = True
    masks = masks.view(batch_size, max_length)

    return batch_tensor, masks

def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, inf=1e10):
    """Radius search in pack mode (fast version).

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.
        inf (float=1e10): infinity value.

    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """
    # pack to batch
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)  # (B, M', 3)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)  # (B, N', 3)
    # knn
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, neighbor_limit)  # (B, M', K)
    # accumulate index
    batch_start_index = torch.cumsum(s_lengths, dim=0) - s_lengths
    batch_knn_indices += batch_start_index.view(-1, 1, 1)
    batch_knn_masks = torch.gt(batch_knn_distances, radius)
    batch_knn_indices.masked_fill_(batch_knn_masks, s_points.shape[0])  # (B, M', K)
    # batch to pack
    knn_indices, _ = batch_to_pack(batch_knn_indices, batch_q_masks)  # (M, K)
    return knn_indices

def keops_radius_count(q_points, s_points, radius):
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    vij = (radius - dij).relu().sign()  # (*, N, M)
    radius_counts = vij.sum(dim=num_batch_dims + 1)  # (*, N)
    return radius_counts