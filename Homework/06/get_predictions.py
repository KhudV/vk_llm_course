import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def predict_by_token_id(logits: torch.Tensor, tokenizer: AutoTokenizer) -> int:
    """
    Determines the predicted choice based on the logits of the model's output.

    Args:
        logits (torch.Tensor): The logits output from the model, typically of shape (1, sequence_length, vocab_size).
        tokenizer (AutoTokenizer): The tokenizer used to encode the input prompt.

    Returns:
        int: The index of the predicted choice (0 for 'A', 1 for 'B', 2 for 'C', 3 for 'D').
    """
    last_token_logits = logits[0, -1, :]
    answer_tokens = ['A', 'B', 'C', 'D']
    token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in answer_tokens]

    answer_logits = last_token_logits[token_ids]
    predicted_choice = torch.argmax(answer_logits).item()
    return predicted_choice


def get_choice_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """
    Calculates the average log probabilities of predicted tokens for a given sequence.


    Args:
        logits (torch.Tensor): A tensor of logits generated by the model, with shape (batch_size, sequence_length, vocab_size).
        input_ids (torch.Tensor): A tensor of input token IDs, with shape (batch_size, sequence_length).

    Returns:
         float: The average log probability of the predicted tokens.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    shifted_input_ids = input_ids[:, 1:]
    shifted_log_probs = log_probs[:, :-1, :]
    target_log_probs = torch.gather(shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    average_log_prob = target_log_probs.mean().item()

    return average_log_prob
