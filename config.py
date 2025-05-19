from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class RewardConfig:
    cache_dir: str
    dataset_path: str
    policy_model_name: str
    policy_model_save_path: str 
    scibert_model_name: str
    scibert_weight_path: str
    enc_model_name: str
    enc_weight_path: str
    reward_lambda: float
    custom_loss_type : str

    @classmethod
    def load(cls, path: str):
        cfg = OmegaConf.load(path)
        return cls(
            cache_dir=cfg.cache_dir,
            dataset_path=cfg.dataset_path,
            policy_model_name=cfg.policy_model_name,
            policy_model_save_path=cfg.policy_model_save_path,
            scibert_model_name=cfg.scibert.model_name,
            scibert_weight_path=cfg.scibert.weight_path,
            enc_model_name=cfg.encoder.model_name,
            enc_weight_path=cfg.encoder.weight_path,
            reward_lambda=cfg.reward_lambda,
            custom_loss_type=cfg.custom_loss_type
        )