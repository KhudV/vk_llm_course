import torch


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
    _, kv_seq_len, num_kv_heads, _ = key.shape

    scale = hidden_dim ** -0.5

    # Group heads to share the key and value if necessary
    if num_heads % num_kv_heads != 0:
        raise ValueError("Number of heads should be divisible by number of key-value heads.")

    heads_per_kv_head = num_heads // num_kv_heads

    # Reshape query for grouped attention
    query = query.view(batch_size, seq_len, num_kv_heads, heads_per_kv_head, hidden_dim)
    key = key.unsqueeze(3)  # Add dimension for heads_per_kv_head
    value = value.unsqueeze(3)

    # Compute scaled dot product attention scores
    attn_scores = torch.einsum("bthqd,btkqd->bthqk", query, key) * scale

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.tril(torch.ones(seq_len, kv_seq_len, device=query.device)).bool()
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    # Compute attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)

    attn_output = torch.einsum("bthqk,btkqd->bthqd", attn_weights, value)
    attn_output = attn_output.reshape(batch_size, seq_len, num_heads, hidden_dim)

    attn_weights = attn_weights.reshape(batch_size, num_heads, seq_len, kv_seq_len)

    if need_weights:
        return attn_output, attn_weights
    else:
        return attn_output, None