# GRPO Training for Qwen on POLARIS Dataset

This repository contains code to train a Qwen 4B model using the Group Relative Policy Optimization (GRPO) algorithm from the TRL (Transformer Reinforcement Learning) library on the POLARIS-Project/Polaris-Dataset-53K dataset.

## Overview

- **Model**: Qwen 2.5 3B (4B parameter variant)
- **Dataset**: POLARIS-Project/Polaris-Dataset-53K
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Library**: TRL from HuggingFace

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- Git

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd grpo
```

### 2. Create a virtual environment

```bash
# Using conda
conda create -n qwen-grpo python=3.10
conda activate qwen-grpo

# Or using venv
python -m venv qwen-grpo
source qwen-grpo/bin/activate  # On Windows: qwen-grpo\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install DeepSpeed for faster training

```bash
pip install deepspeed
```

### 5. (Optional) Set up WandB for experiment tracking

```bash
wandb login
```

## Project Structure

```
grpo/
├── requirements.txt          # Python dependencies
├── config.py               # Training configuration
├── preprocess_data.py      # Dataset preprocessing script
├── train_grpo.py           # Main training script
└── README.md               # This file
```

## Configuration

The `config.py` file contains all hyperparameters and settings for training. Key sections:

### Model Configuration
- Model selection (Qwen 3B, 7B, etc.)
- Quantization settings (8-bit, 4-bit)
- Precision (FP16, BF16)

### Training Configuration
- Learning rate, batch size, epochs
- Optimizer settings
- Sequence lengths

### GRPO Configuration
- PPO clipping parameter (epsilon)
- Discount factor (gamma)
- GAE parameter (gae_lambda)
- KL penalty coefficient

### Hardware Configuration
- GPU settings
- DeepSpeed configuration
- Multi-GPU training

Modify `config.py` before training to adjust these parameters.

## Usage

### Step 1: Explore and Preprocess the Dataset

Before training, it's recommended to explore the dataset structure:

```bash
python preprocess_data.py
```

This will:
- Load and display dataset information
- Show sample examples
- Save 100 examples to `sample_polaris_data.json` for inspection
- Format the dataset for GRPO training

**Important**: Review `sample_polaris_data.json` to understand the dataset structure and adjust field names in `config.py` if necessary.

### Step 2: Train the Model

Run the main training script:

```bash
python train_grpo.py
```

#### Training Options

**Single GPU Training (default)**:
```bash
python train_grpo.py
```

**Multi-GPU Training**:
```bash
accelerate launch --multi_gpu --num_processes=4 train_grpo.py
```

**With DeepSpeed**:
```bash
accelerate launch --deepspeed deepspeed_config.json train_grpo.py
```

**With Quantization (8-bit)**:
Edit `config.py` and set `load_in_8bit = True` in `MODEL_CONFIG`, then run:
```bash
python train_grpo.py
```

**With QLoRA (Parameter-Efficient Fine-Tuning)**:
Edit `config.py` and set `use_lora = True` in `ADVANCED_CONFIG`, then run:
```bash
python train_grpo.py
```

### Step 3: Monitor Training

#### Using WandB
If you've set up WandB, training metrics will be automatically logged:
```bash
wandb dashboard
```

#### Using TensorBoard
```bash
tensorboard --logdir=./logs/qwen-grpo-polaris
```

Then open `http://localhost:6006` in your browser.

## Customizing the Reward Function

The default reward function in `train_grpo.py` is a simple length-based reward. To customize it for your specific task:

1. Open `train_grpo.py`
2. Find the `compute_reward_function` function
3. Modify the reward calculation logic

Example: Keyword-based reward
```python
def compute_reward_function(completions, **kwargs):
    rewards = []
    positive_keywords = ["correct", "yes", "true"]
    negative_keywords = ["wrong", "no", "false"]
    
    for completion in completions:
        reward = 0.5  # Base reward
        completion_lower = completion.lower()
        
        for keyword in positive_keywords:
            if keyword in completion_lower:
                reward += 0.3
        
        for keyword in negative_keywords:
            if keyword in completion_lower:
                reward -= 0.3
        
        rewards.append(max(0, min(1, reward)))  # Clamp between 0 and 1
    
    return rewards
```

## Output Files

After training, the following directories will be created:

- `./output/qwen-grpo-polaris/` - Trained model checkpoints
- `./logs/qwen-grpo-polaris/` - Training logs and metrics
- `./sample_polaris_data.json` - Sample data for inspection (after preprocessing)

## Memory Requirements

| Configuration | GPU Memory Required |
|--------------|---------------------|
| Full precision (FP32) | ~48 GB |
| Mixed precision (FP16) | ~24 GB |
| 8-bit quantization | ~12 GB |
| 4-bit quantization | ~8 GB |
| QLoRA (4-bit) | ~6 GB |

## Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**: Set `batch_size = 1` in `config.py`
2. **Enable gradient checkpointing**: Already enabled by default
3. **Use quantization**: Set `load_in_8bit = True` or `load_in_4bit = True`
4. **Use QLoRA**: Set `use_lora = True` and `load_in_4bit = True`
5. **Reduce sequence length**: Decrease `max_length` in `config.py`

### Slow Training

1. **Use DeepSpeed**: Enable DeepSpeed configuration
2. **Increase gradient accumulation**: Increase `gradient_accumulation_steps`
3. **Use multiple GPUs**: Run with `accelerate launch --multi_gpu`
4. **Enable mixed precision**: Already enabled by default (FP16)

### Dataset Loading Issues

1. **Check internet connection**: Ensure you can access HuggingFace Hub
2. **Verify dataset name**: Check `dataset_path` in `config.py`
3. **Review dataset structure**: Run `preprocess_data.py` to inspect the dataset

## Performance Tips

1. **Start small**: Test with a subset of data first by setting `max_train_samples = 1000` in `config.py`
2. **Monitor metrics**: Use WandB or TensorBoard to track training progress
3. **Adjust learning rate**: If training diverges, try reducing `learning_rate` by 10x
4. **Use warmup**: The default `warmup_ratio = 0.1` helps with stable training
5. **Save checkpoints**: Regular checkpoints (every 500 steps) allow resuming from failures

## Advanced Usage

### Custom Reward Model

To use a custom reward model instead of a heuristic function:

1. Train or download a reward model
2. Set `reward_model_path` in `config.py`
3. Modify `train_grpo.py` to load and use the reward model

### Distributed Training

For multi-node training:

```bash
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=8 \
    --num_machines=2 \
    --main_process_port=29500 \
    train_grpo.py
```

### Resume Training

To resume from a checkpoint:

```bash
python train_grpo.py --resume_from_checkpoint ./output/qwen-grpo-polaris/checkpoint-1000
```

## Citation

If you use this code, please cite:

```bibtex
@software{qwen_grpo_polaris,
  title={GRPO Training for Qwen on POLARIS Dataset},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/grpo}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen model
- [TRL Library](https://github.com/huggingface/trl) for the GRPO implementation
- [POLARIS Project](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K) for the dataset

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This code is provided as-is for research purposes. Always verify the dataset license and model usage terms before training or deploying.