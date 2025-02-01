import torch
import math


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
              Only returned if 'need_weights' is True.
    """
    batch_size, seq_len, num_heads, hidden_dim = query.shape
    batch_size_kv, kv_seq_len, num_kv_heads, hidden_dim_kv = key.shape

    if num_kv_heads > num_heads or (num_heads % num_kv_heads) != 0:
        raise ValueError("Number of key/value heads must be <= number of query heads and divide it exactly.")

    group_factor = num_heads // num_kv_heads

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    k = k.repeat_interleave(group_factor, dim=1)
    v = v.repeat_interleave(group_factor, dim=1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hidden_dim)

    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, kv_seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)

    output = torch.matmul(attn_weights, v)

    output = output.permute(0, 2, 1, 3)
    if need_weights:
        return output, attn_weights
    else:
        return output, None



