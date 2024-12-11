from scripts.compute_reward import compute_reward
from torch import no_grad



def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """

    # <YOUR CODE HERE>
    input_text = "1"
    inputs = main_tokenizer(
        input_text, 
        return_tensors='pt', 
    ).to(device)

    generated_texts = []
    for _ in range(N):
        with no_grad():
            output = main_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
            )
        decoded_text = main_tokenizer.decode(output[0])
        if decoded_text.startswith("tensor(") and decoded_text.endswith(")"):
            decoded_text = decoded_text[7:-1]
        generated_texts.append(decoded_text)

    reward_scores = compute_reward(reward_model, reward_tokenizer, generated_texts)

    best_index = reward_scores.argmax().item()
    best_text = generated_texts[best_index]
    
    return best_text