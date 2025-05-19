import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Any, Union, Literal
from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import (
 RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)
from config import RewardConfig

@dataclass
class CustomCollator(DataCollatorMixin):
  pad_token_id: int
  return_tensors: str = "pt"

  def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
    # Convert to tensor
    prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
    prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
    chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
    chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
    rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
    rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
    if "pixel_values" in examples[0]:
      pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
    if "pixel_attention_mask" in examples[0]:
      pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
    if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
      ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
      ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

    # Pad
    output = {}
    output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
    output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
    output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
    output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
    output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
    output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
    if "pixel_values" in examples[0]:
      output["pixel_values"] = pad(pixel_values, padding_value=0.0)
    if "pixel_attention_mask" in examples[0]:
      output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
    if "image_sizes" in examples[0]:
      output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
    if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
      output["ref_chosen_logps"] = ref_chosen_logps
      output["ref_rejected_logps"] = ref_rejected_logps
    if "chosen_inputs" in examples[0]:
      chosen_inputs = torch.tensor([example["chosen_inputs"] for example in examples])
      output["chosen_inputs"] = chosen_inputs

    if "rejected_inputs" in examples[0]:
      rejected_inputs = torch.tensor([example["rejected_inputs"] for example in examples])
      output["rejected_inputs"] = rejected_inputs

    if "chosen_margin" in examples[0]:
      chosen_margin = torch.tensor([example["chosen_margin"] for example in examples])
      output["chosen_margin"] = chosen_margin

    if "rejected_margin" in examples[0]:
      rejected_margin = torch.tensor([example["rejected_margin"] for example in examples])
      output["rejected_margin"] = rejected_margin

    return output




class CustomDPOTrainer(DPOTrainer):
  def __init__(self, reward_model: nn.Module, encoder_model: nn.Module, *args: Any, **kwargs: Any):
    self.reward_model = reward_model
    self.enc_model = encoder_model
    super().__init__(*args, **kwargs)

  def set_custom_config(self, config):
    self.custom_config = RewardConfig.load(config)
    print(self.custom_config.custom_loss_type)
      
  def _set_signature_columns_if_needed(self):
    # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
    # By default, this method sets `self._signature_columns` to the model's expected inputs.
    # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
    # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
    if self._signature_columns is None:
      self._signature_columns = [
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "image_sizes",
        "ref_chosen_logps",
        "ref_rejected_logps",
        "chosen_inputs",
        "rejected_inputs",
        "chosen_margin",
        "rejected_margin",
        "label"
      ]

  def tokenize_row(self, features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
    """
    Tokenize a row of the dataset.

    """
    tokenizer = processing_class  # the processing class is a tokenizer
    prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    ### custom
    chosen_inputs = self.enc_model.encode(features["chosen"]).squeeze(0)
    rejected_inputs = self.enc_model.encode(features["rejected"]).squeeze(0)

    chosen_margin = self.reward_model.margin_score(features["chosen"], features['label'])
    rejected_margin = self.reward_model.margin_score(features["rejected"], features['label'])

    # Add special tokens (typically for encoder-decoder models)
    if add_special_tokens:
      if tokenizer.bos_token_id is not None:
        prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
      if tokenizer.eos_token_id is not None:
        prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    # Truncate prompt and completion sequences
    if max_prompt_length is not None:
      prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
      chosen_input_ids = chosen_input_ids[:max_completion_length]
      rejected_input_ids = rejected_input_ids[:max_completion_length]

    return {
      "prompt_input_ids": prompt_input_ids,
      "chosen_input_ids": chosen_input_ids,
      "rejected_input_ids": rejected_input_ids,
      "chosen_inputs": chosen_inputs.cpu().tolist(),  # enc_model tensors
      "rejected_inputs": rejected_inputs.cpu().tolist(),  #enc_model tensors
      "chosen_margin": chosen_margin,   # roberta model margin
      "rejected_margin": rejected_margin
    }

  ## DPO loss override
  def dpo_loss(
      self,
      chosen_logps: torch.FloatTensor,
      rejected_logps: torch.FloatTensor,
      ref_chosen_logps: torch.FloatTensor,
      ref_rejected_logps: torch.FloatTensor,
      enc_chosen_hidden: torch.FloatTensor,
      enc_rejected_hidden : torch.FloatTensor,
      chosen_rm_rewards: torch.FloatTensor,
      rejected_rm_rewards: torch.FloatTensor,
      alpha: float = 0.5,  # DPO-RM utility weighting factor
  ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    utility DPO loss function combining original DPO with RM-based ranking loss.

    Args:
        chosen_logps: (B,) tensor of model log-probs for chosen samples
        rejected_logps: (B,) tensor of model log-probs for rejected samples
        ref_chosen_logps: (B,) tensor of reference model log-probs for chosen samples
        ref_rejected_logps: (B,) tensor of reference model log-probs for rejected samples
        chosen_rm_rewards: (B,) tensor of RM scores for chosen samples
        rejected_rm_rewards: (B,) tensor of RM scores for rejected samples
        alpha: weighting between DPO and RM-based losses

    Returns:
        total_losses, chosen_rewards, rejected_rewards, rm_based_losses
    """
    tanh = nn.Tanh()
    device = self.accelerator.device

    # ---------------------------------------
    # DPO LOSS (Standard DPO log-ratio logic)
    # ---------------------------------------
    logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = logratios - ref_logratios

    if self.loss_type == "sigmoid":
      dpo_losses = (
          -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
          - F.logsigmoid(-self.beta * logits) * self.label_smoothing
      )
    else:
      raise ValueError("Only sigmoid supported for utility loss.")

    chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)

    ###############
    rp, rn = self.enc_model.r_util(enc_chosen_hidden, chosen_rm_rewards), self.enc_model.r_util(enc_rejected_hidden, rejected_rm_rewards)
    ranking_loss = tanh(torch.abs(rp - rn))
      
    rm_based_losses = ranking_loss.to(device)
    # ---------------------------------------
    # RM BASED LOSS (Ranking loss with RM scores)
    # ---------------------------------------

    if self.custom_config.custom_loss_type == "based_dpo":
      return dpo_losses, chosen_rewards, rejected_rewards, rm_based_losses

    elif self.custom_config.custom_loss_type == "utility":
      total_losses = dpo_losses * rm_based_losses
      #print(total_losses, chosen_rewards, rejected_rewards, rm_based_losses)
      return total_losses, chosen_rewards, rejected_rewards, rm_based_losses

    else:
      return dpo_losses, chosen_rewards, rejected_rewards, rm_based_losses
    # ---------------------------------------
    # utility LOSS (Combine DPO + RM)
    # ---------------------------------------

  def get_batch_loss_metrics(
      self,
      model,
      batch: dict[str, Union[list, torch.LongTensor]],
      train_eval: Literal["train", "eval"] = "train",
  ):
    device = self.accelerator.device
    # ────────────────────────────────────────────────────────────────────
    # 1)  Forward pass through policy / ref to get standard DPO signals
    # ────────────────────────────────────────────────────────────────────
    metrics = {}

    model_output = self.concatenated_forward(model, batch)
    if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
      ref_chosen_logps = batch["ref_chosen_logps"]
      ref_rejected_logps = batch["ref_rejected_logps"]
    else:
      ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

    # ────────────────────────────────────────────────────────────────────
    # 2)  External RM forward (no‑grad) to compute current margin signal
    # ────────────────────────────────────────────────────────────────────

    with torch.no_grad():
      device = self.accelerator.device

      # Retrieve per‑sample margins from batch (fallback to zeros if
      # not provided so code doesn’t crash during sanity checks).
      chosen_inputs = batch.get("chosen_inputs", torch.zeros_like(ref_chosen_logps))
      rejected_inputs = batch.get("rejected_inputs", torch.zeros_like(ref_chosen_logps))
      chosen_margin = batch.get("chosen_margin", torch.zeros_like(ref_chosen_logps))  # batch 에서 선택한 positive maragin
      rejected_margin = batch.get("rejected_margin",
                                  torch.zeros_like(ref_rejected_logps))  # batch 에서 선택한 negative margin


    ### dpo loss 함수 호출
    losses, chosen_rewards, rejected_rewards, rm_based_loss = self.dpo_loss(
      model_output["chosen_logps"],
      model_output["rejected_logps"],
      ref_chosen_logps,
      ref_rejected_logps,
      chosen_inputs,
      rejected_inputs,
      chosen_margin,
      rejected_margin
    )

    # ────────────────────────────────────────────────────────────────────
    # 3)  Collect metrics (unchanged + new rm_margin entry)
    # ────────────────────────────────────────────────────────────────────
    metrics = {}
    prefix = "eval_" if train_eval == "eval" else ""

    metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
    metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
    metrics[f"{prefix}rewards/accuracies"] = (
      self.accelerator.gather_for_metrics((chosen_rewards > rejected_rewards).float()).mean().item()
    )
    metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(
      chosen_rewards - rejected_rewards
    ).mean().item()
    metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(model_output["chosen_logps"]).mean().item()
    metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(
      model_output["rejected_logps"]).mean().item()
    metrics[f"{prefix}rm/margin"] = rm_based_loss.mean()  # ← NEW: external RM margin

    # print(metrics)
    return losses.mean(), metrics
