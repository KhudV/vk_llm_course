def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    subject = sample["subject"]
    question = sample["question"]
    options = sample["choices"]

    prompt = (
        f"The following are multiple choice questions (with answers) about {subject}.\n"
        f"{question}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:"
    )

    return prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    formatted_examples = []
    for example in examples:
        subject = example["subject"]
        question = example["question"]
        options = example["choices"]
        answer_index = example["answer"]
        correct_letter = chr(ord('A') + answer_index)

        if add_full_example:
            correct_answer = options[answer_index]
            formatted_examples.append(
                f"The following are multiple choice questions (with answers) about {subject}.\n"
                f"{question}\n"
                f"A. {options[0]}\n"
                f"B. {options[1]}\n"
                f"C. {options[2]}\n"
                f"D. {options[3]}\n"
                f"Answer: {correct_letter}. {correct_answer}"
            )
        else:
            formatted_examples.append(
                f"The following are multiple choice questions (with answers) about {subject}.\n"
                f"{question}\n"
                f"A. {options[0]}\n"
                f"B. {options[1]}\n"
                f"C. {options[2]}\n"
                f"D. {options[3]}\n"
                f"Answer: {correct_letter}"
            )

    formatted_examples_str = "\n\n".join(formatted_examples)

    sample_prompt = (
        f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
        f"{sample['question']}\n"
        f"A. {sample['choices'][0]}\n"
        f"B. {sample['choices'][1]}\n"
        f"C. {sample['choices'][2]}\n"
        f"D. {sample['choices'][3]}\n"
        f"Answer:"
    )

    return f"{formatted_examples_str}\n\n{sample_prompt}"