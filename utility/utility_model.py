import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from config import RewardConfig

class SciBERTRewardsModel:
    def __init__(self, config, num_labels, device):
        self.custom_config = RewardConfig.load(config)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.custom_config.scibert_model_name, cache_dir=self.custom_config.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.custom_config.scibert_model_name, num_labels=num_labels, cache_dir=self.custom_config.cache_dir).to(self.device)
        if self.custom_config.scibert_weight_path is None:
            print("[Utility Model] No weight loaded; using randomly initialized.")
        else:
            self.model.load_state_dict(torch.load(self.custom_config.scibert_weight_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def margin_score(self, text: str, label: int) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        logits = self.model(**inputs).logits.squeeze(0)

        true_score = logits[label]
        wrong_score = logits.masked_fill(torch.nn.functional.one_hot(torch.tensor(label), logits.size(-1)).bool().to(self.device), -1e9).max()

        return (true_score - wrong_score).item()


class WHead(nn.Module):
    def __init__(self, config ,device):
        super().__init__()
        self.custom_config = RewardConfig.load(config)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.custom_config.enc_model_name, cache_dir=self.custom_config.cache_dir)
        self.encoder = AutoModel.from_pretrained(self.custom_config.enc_model_name, cache_dir=self.custom_config.cache_dir).to(self.device)
        self.encoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.w_head = nn.Linear(self.encoder.config.hidden_size, 1, bias=True).to(self.device)
        if self.custom_config.enc_weight_path is None:
            print("[WHead] No weight loaded; using randomly initialized head.")
        else:
            self.w_head.load_state_dict(torch.load(self.custom_config.enc_weight_path, map_location=self.device))
        self.w_head.eval()

        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        hidden = self.encoder(**tokens).last_hidden_state[:, 0]
        return hidden

    @torch.no_grad()
    def r_util(self, vector: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.w_head(vector)).squeeze(-1) + self.custom_config.reward_lambda * margin.to(self.device)
