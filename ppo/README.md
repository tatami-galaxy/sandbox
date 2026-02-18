# PPO Training for Qwen on POLARIS Dataset

This repository contains code to train a Qwen 4B model on the POLARIS-Dataset-53K using the Proximal Policy Optimization (PPO) algorithm with the TRL library from Hugging Face.

## Overview

- **Model**: Qwen 4B (Qwen/Qwen2.5-4B)
- **Dataset**: POLARIS-Project/Polaris-Dataset-53K
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Library**: TRL (Transformer Reinforcement Learning)
- **Training Method**: LoRA (Low-Rank Adaptation) for parameter-efficient training

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for 4B model)
- 20GB+ RAM

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Weights & Biases for experiment tracking:
```bash
pip install wandb
wandb login
```

## Usage

### Basic Training

Two versions are available:

1. **Simplified version (recommended)** - Uses PPOTrainer directly:
```bash
python train_ppo_simple.py
```

2. **Full version with wrapper** - Includes additional helper methods:
```bash
python train_ppo.py
```

This will:
- Load Qwen 2.5-1.5B model (smaller for faster training)
- Apply LoRA adapters
- Load POLARIS dataset
- Train with PPO for 20 epochs
- Save checkpoints every 5 epochs
- Save final model to `qwen_ppo_polaris_final/`

### Using Configuration Presets

Edit `train_ppo.py` to use different configurations:

```python
# For low-memory environments (smaller model, 4-bit quantization)
config = get_low_memory_config()

# For higher quality training (more epochs, reward model)
config = get_high_quality_config()
```

### Custom Configuration

Modify the configuration parameters in `config.py`:

```python
from config import TrainingConfig, ModelConfig, PPOTrainingConfig

config = TrainingConfig()
config.model.model_name = "Qwen/Qwen2.5-4B"  # Use 4B model
config.ppo.batch_size = 8  # Larger batch size
config.ppo.total_ppo_epochs = 50  # More epochs
config.reward.reward_type = "model"  # Use reward model
```

## Configuration Options

### Model Configuration

- `model_name`: Hugging Face model ID (default: "Qwen/Qwen2.5-4B")
- `use_4bit`: Enable 4-bit quantization for memory efficiency
- `use_lora`: Enable LoRA for parameter-efficient training
- `lora_r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha (default: 32)

### Training Configuration

- `learning_rate`: Learning rate (default: 1.41e-5)
- `batch_size`: Batch size for training (default: 4)
- `total_ppo_epochs`: Number of PPO epochs (default: 20)
- `ppo_epochs`: PPO epochs per batch (default: 4)
- `max_new_tokens`: Maximum tokens to generate (default: 256)
- `temperature`: Sampling temperature (default: 0.7)

### Reward Configuration

- `reward_type`: "heuristic", "model", or "hybrid"
- `reward_model_name`: Reward model ID for model-based rewards
- `heuristic_weight`: Weight for heuristic rewards (in hybrid mode)

## Memory Requirements

| Model | Batch Size | VRAM Required | Notes |
|-------|------------|---------------|-------|
| Qwen 1.5B (4-bit) | 2 | ~8GB | Low memory mode |
| Qwen 1.5B | 4 | ~12GB | Default configuration |
| Qwen 4B | 2 | ~16GB | Recommended for 4B |
| Qwen 4B | 4 | ~24GB | Higher quality |

## Output Files

- `checkpoints/checkpoint_epoch_N/`: Intermediate checkpoints
- `qwen_ppo_polaris_final/`: Final trained model
- Logs are saved to Weights & Biases if configured

## Using the Trained Model

Load and use the trained model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-4B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "qwen_ppo_polaris_final")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("qwen_ppo_polaris_final")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Advanced: Custom Reward Model

For better results, use a trained reward model instead of heuristic rewards:

1. Create a `reward_model.py` file (see `reward_model_example.py` for reference)
2. Set `reward.reward_type = "model"` in configuration
3. Provide `reward.reward_model_name`

## Monitoring Training

### Weights & Biases

The training script logs to Weights & Biases by default. View your training at:
```
https://wandb.ai/<your-entity>/qwen-ppo-polaris
```

### TensorBoard

To use TensorBoard instead, modify configuration:
```python
config.ppo.log_with = "tensorboard"
```

Then view:
```bash
tensorboard --logdir ./runs
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` and `mini_batch_size`
- Enable 4-bit quantization: `model.use_4bit = True`
- Use smaller model: `model.model_name = "Qwen/Qwen2.5-1.5B"`
- Reduce `max_new_tokens`

### Slow Training

- Increase `batch_size` if memory allows
- Reduce `ppo_epochs` per batch
- Use smaller model for initial experiments

### Poor Results

- Increase `total_ppo_epochs`
- Use a better reward model
- Adjust `learning_rate`
- Increase `max_new_tokens` for longer responses

## Dataset Information

The POLARIS-Dataset-53K contains 53,000 examples for training. The script automatically loads the dataset and adapts to its structure. You may need to adjust field mappings in `DatasetConfig` if the dataset structure changes.

## Citation

If you use this code, please cite:

```bibtex
@software{trl,
  author = {von Werra, Leandro and Younes, Edouard and others},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/huggingface/trl}
}
```

## License

This code is provided as-is for educational and research purposes. Please respect the licenses of the underlying models and datasets.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Contact

For questions or issues, please open a GitHub issue.