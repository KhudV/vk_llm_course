import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    hidden_dim = queries.size(-1)
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_weights, values)
    return attention_output




def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD = queries.size()
    
    attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(DIM_PER_HEAD, dtype=torch.float32))
    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_output = torch.matmul(attention_weights, values)
    
    concatenated_output = attention_output.permute(0, 2, 1, 3).reshape(BATCH_SIZE, SEQ_LENGTH, N_HEADS * DIM_PER_HEAD)
    output = torch.matmul(concatenated_output, projection_matrix.T)
    
    return output




def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = x.shape

    m = torch.arange(SEQ_LENGTH, dtype=torch.float32, device=x.device)
    i = torch.arange(DIM_PER_HEAD // 2, dtype=torch.float32, device=x.device)
    theta = 10000 ** (-2 * i / DIM_PER_HEAD)
    
    m_theta = m[:, None] * theta[None, :]
    cos_theta = torch.cos(m_theta)
    sin_theta = torch.sin(m_theta)
    
    cos_theta = cos_theta[None, :, None, :].repeat(BATCH_SIZE, 1, N_HEADS, 1)
    sin_theta = sin_theta[None, :, None, :].repeat(BATCH_SIZE, 1, N_HEADS, 1)

    x1, x2 = x[..., ::2], x[..., 1::2]

    x_rotated = torch.stack(
        [x1 * cos_theta - x2 * sin_theta, x2 * cos_theta + x1 * sin_theta], dim=-1
    ).flatten(-2)
    
    return x_rotated
