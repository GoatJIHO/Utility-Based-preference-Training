import os
import argparse
import torch
import numpy as np
import random
import gc
from tqdm import tqdm
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from trl import DPOTrainer, DPOConfig

from config import RewardConfig
from utility.utility_model import SciBERTRewardsModel, WHead
from trainer.hybrid_dpo_trainer import CustomCollator, CustomDPOTrainer
from trainer.pair_generator import pair_generator



def main(config_path):
    # -------------------- Load Config --------------------
    cfg = RewardConfig.load(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------- Load Tokenizer & Model --------------------
    policy_tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_name, cache_dir=cfg.cache_dir)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_model_name, 
        cache_dir=cfg.cache_dir,
        use_cache=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    ).to(device)

    # -------------------- Load Dataset --------------------
    dataset = load_from_disk(cfg.dataset_path)
    num_labels = len(set(dataset["label"]))

    # -------------------- Reward Models --------------------
    utility_model = SciBERTRewardsModel(config_path, num_labels, device)
    w_head_model = WHead(config_path, device)

   
    # -------------------- Generate data --------------------
    policy_model.eval()
    generated_dataset = pair_generator(
        policy_tokenizer= policy_tokenizer,
        policy_model = policy_model,
        utility_model= utility_model, 
        train_dataset =dataset['train'], 
        iters=200, 
        resampling_n=5,
        num_labels=num_labels
        )

    # -------------------- Collator & Trainer --------------------
    policy_model.train()
    collator = CustomCollator(pad_token_id=policy_tokenizer.pad_token_id)

    dpo_config = DPOConfig(
        beta=1.0,
        learning_rate=2e-5,
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3
    )

    trainer = CustomDPOTrainer(
        reward_model=utility_model,
        encoder_model=w_head_model,
        processing_class=policy_tokenizer,
        model=policy_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=generated_dataset,
        data_collator=collator
    )

    trainer.set_custom_config(config_path)
    trainer.train()

    policy_model_save = cfg.policy_model_save_path

    policy_model.save_pretrained(policy_model_save)
    policy_tokenizer.save_pretrained(policy_model_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO training with config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    main(args.config_path)
