# Utility-Based Preference Optimization for Synthetic Text Classification

This project implements utility-guided preference training for synthetic text generation and classification, using DPO-style optimization with reward modeling.

---

## 📁 Dataset Format

Your dataset should be stored as a Hugging Face `datasets.DatasetDict` with at least the following structure:

```python
{
  'train': Dataset({
      'text': [...],     # Input text or abstract
      'label': [...]     # Corresponding class labels (int)
  })
}
```


Place the dataset directory inside the data/ folder.

You can use Dataset.from_dict(...) or Dataset.from_pandas(...) to prepare it.

The dataset will be loaded using load_from_disk(cfg.dataset_path).


⚙️ Configuration (config.yaml)
The experiment is driven by a single YAML config file. A minimal example looks like this:
```
cache_dir: "./model_cache"
dataset_path: "./data/your_dataset"
policy_model_save_path: "./models/policy_model"

scibert:
  model_name: "allenai/scibert_scivocab_uncased"
  weight_path: null  # Set to null if no fine-tuned weights are available

encoder:
  model_name: "microsoft/MiniLM-L12-H384-uncased"
  weight_path: null  # Set to null if no fine-tuned weights are available

policy:
  model_name: "microsoft/Phi-4-mini-instruct"

reward_lambda: 0.1
custom_loss_type: "hybrid"
```
- If no fine-tuned model is available, set the corresponding weight_path to null.

- policy.model_name is the LLM used to generate synthetic abstracts.

🚀 Running the Training
Run the main training script with:

```
python main.py --config_path ./config.yaml
```
This script will:

- Load the dataset and config

- Generate synthetic prompt-response pairs via margin-based sampling

- Train the policy model using Direct Preference Optimization (DPO)

