import gc
import torch
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm


def build_prompt(sample_list) -> str: 
    prompt = """
    You are an expert academic assistant. The following Examples are academic paper abstracts in the field of something
    They are written in formal scientific style.

    Your task is to generate a new academic abstract in a similar style and topic.

    Examples:
    ---
    """
    prompt += f"Example 1:\n{sample_list[0][:700].strip()}\n\n"
    prompt += f"Example 2:\n{sample_list[1][:700].strip()}\n\n"
    prompt += """
    Now, generate a single academic abstract paragraph in the same domain.
    Only output the abstract content. Do not include titles, citations, links, or additional instructions.

    Abstract: 
    """
    return prompt

def generate_batch_prompts(label_to_texts, label_list):
    prompts = []
    for y_label in label_list:
        sample_data = random.sample(label_to_texts[y_label], k=2)
        prompt = build_prompt(sample_data)
        prompts.append(prompt)
    return prompts

def generate_synthetic_batch(policy_tokenizer, policy_model, prompts, n):
    inputs = policy_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    outputs = policy_model.generate(
        **inputs,
        max_new_tokens=512,
        top_p=0.95,
        top_k=50,
        temperature=1.0,
        do_sample=True,
        num_return_sequences=n,
        pad_token_id=policy_tokenizer.eos_token_id
    )

    prompt_lens = (inputs['input_ids'] != policy_tokenizer.pad_token_id).sum(dim=1)

    total = len(prompts) * n
    decoded = policy_tokenizer.batch_decode(outputs[:, prompt_lens[0]:], skip_special_tokens=True)

    return decoded  # length: total

def pair_generator(policy_tokenizer, policy_model, utility_model, train_dataset  , iters: int, resampling_n: int, num_labels: int ,batch_size: int = 8) -> list:
    label_counter = defaultdict(int)
    label_to_texts = defaultdict(list)

    for idx in range(num_labels): label_counter[idx]
    for example in train_dataset: label_to_texts[example['label']].append(example['text'])

    pairs = []
    label_counter = defaultdict(int)
    pbar = tqdm(range(num_labels), desc="pair generating")

    for y_label in pbar:
        while label_counter[y_label] < iters:
            current_batch_size = min(batch_size, iters - label_counter[y_label])
            print(current_batch_size)
            batch_labels = [y_label] * current_batch_size
            prompts = generate_batch_prompts(policy_tokenizer, policy_model, label_to_texts, batch_labels)  # B prompts

            # total generated texts = B * n
            all_texts = generate_synthetic_batch(prompts, resampling_n)

            # margin score for all (flattened)
            all_labels = [y_label] * len(all_texts)
            margins = scibert_model.margin_score(all_texts, all_labels)

            # regroup per-prompt basis
            for i in range(current_batch_size):
                start_idx = i * resampling_n
                end_idx = (i + 1) * resampling_n
                m = margins[start_idx:end_idx]
                texts = all_texts[start_idx:end_idx]

                top_idx = m.topk(resampling_n).indices[0].item()
                sorted_neg_indices = m.argsort()
                middle_idx = sorted_neg_indices[len(sorted_neg_indices) // 2].item()

                pairs.append((
                    prompts[i],
                    texts[top_idx],
                    texts[middle_idx],
                    m[top_idx].item(),
                    m[middle_idx].item(),
                    y_label
                ))

            label_counter[y_label] += current_batch_size

            gc.collect()
            torch.cuda.empty_cache()

    return pairs
