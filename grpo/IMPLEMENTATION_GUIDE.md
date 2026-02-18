# Implementation Guide: GRPO Training for Qwen on POLARIS Dataset

This guide provides detailed technical information about implementing GRPO training for Qwen models on the POLARIS dataset.

## Table of Contents
1. [Understanding GRPO](#understanding-grpo)
2. [Architecture Overview](#architecture-overview)
3. [Key Implementation Details](#key-implementation-details)
4. [Dataset Preprocessing](#dataset-preprocessing)
5. [Reward Functions](#reward-functions)
6. [Training Process](#training-process)
7. [Optimization Strategies](#optimization-strategies)
8. [Troubleshooting Guide](#troubleshooting-guide)

## Understanding GRPO

### What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm designed for training language models. It's an extension of PPO (Proximal Policy Optimization) that incorporates group-based relative comparisons.

### Key Concepts

1. **Policy Gradient**: Uses gradient ascent to maximize expected rewards
2. **Clipped Objective**: Prevents large policy updates (similar to PPO)
3. **Group Comparisons**: Compares responses within a group to compute relative advantages
4. **KL Penalty**: Penalizes deviation from the original model distribution

### Mathematical Foundation

The GRPO objective function:

```
L(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)] - β * KL(π_θ || π_ref)
```

Where:
- `r_t(θ)`: Probability ratio (new policy / old policy)
- `A_t`: Advantage estimate using GAE (Generalized Advantage Estimation)
- `ε`: Clipping parameter (default: 0.2)
- `β`: KL penalty coefficient
- `π_θ`: Current policy
- `π_ref`: Reference policy (initial model)

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Dataset     │    │     Model     │    │   Reward      │
│   Loader      │───▶│    (Qwen)     │───▶│  Function     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        │                     ▼                     │
        │            ┌───────────────┐              │
        │            │   GRPO       │◀─────────────┘
        └───────────▶│   Trainer    │
                     └───────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │   Checkpoints │
                     └───────────────┘
```

### Data Flow

1. **Input**: POLARIS dataset with prompts and completions
2. **Processing**: Tokenization and formatting
3. **Generation**: Model generates multiple responses per prompt
4. **Reward Calculation**: Compute rewards for each response
5. **Advantage Estimation**: Calculate advantages using GAE
6. **Policy Update**: Update model using clipped objective
7. **Logging**: Save metrics and checkpoints

## Key Implementation Details

### 1. Model Loading

```python
# Key considerations:
# - Use trust_remote_code=True for Qwen models
# - Set pad_token = eos_token if not exists
# - Use float16 for memory efficiency
# - Enable device_map="auto" for multi-GPU support

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### 2. Dataset Formatting

The dataset must be formatted with `prompt` and `completion` fields:

```python
{
    "prompt": "User: <input text>\nAssistant: ",
    "completion": "<output text>"
}
```

### 3. GRPO Configuration

Critical parameters:

- `epsilon`: PPO clipping (0.1-0.3, default: 0.2)
- `gamma`: Discount factor (0.95-0.99, default: 0.99)
- `gae_lambda`: GAE parameter (0.9-0.99, default: 0.95)
- `kl_penalty`: KL divergence penalty (0.01-0.5, default: 0.1)
- `learning_rate`: Usually lower for RL (1e-7 to 1e-5)

### 4. Memory Optimization

Strategies to reduce memory usage:

1. **Gradient Checkpointing**: Saves intermediate activations
2. **Quantization**: 8-bit or 4-bit weights
3. **QLoRA**: Low-rank adaptation
4. **Mixed Precision**: FP16/BF16 training
5. **Batch Size Reduction**: Smaller batches with gradient accumulation

## Dataset Preprocessing

### POLARIS Dataset Structure

The POLARIS dataset typically contains instruction-following examples. Common field names:
- Input fields: `input`, `question`, `instruction`, `prompt`
- Output fields: `output`, `answer`, `response`, `completion`

### Preprocessing Steps

1. **Load Dataset**: Use HuggingFace datasets library
2. **Explore Structure**: Inspect first few examples
3. **Format Fields**: Map dataset fields to `prompt` and `completion`
4. **Split**: Create train/test/validation splits
5. **Save**: Save to disk for faster loading

### Code Example

```python
from datasets import load_dataset

dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")

def format_example(example):
    prompt = f"User: {example['input']}\nAssistant: "
    completion = example['output']
    return {"prompt": prompt, "completion": completion}

formatted = dataset.map(format_example, remove_columns=dataset.column_names)
```

## Reward Functions

### Types of Reward Functions

#### 1. Length-Based Reward

```python
def length_based_reward(completions):
    rewards = []
    for completion in completions:
        length = len(completion.split())
        reward = min(length / 100.0, 1.0)  # Normalize to [0, 1]
        rewards.append(reward)
    return rewards
```

#### 2. Keyword-Based Reward

```python
def keyword_reward(completions):
    positive_keywords = ["correct", "yes", "true", "accurate"]
    negative_keywords = ["wrong", "no", "false", "error"]
    
    rewards = []
    for completion in completions:
        reward = 0.5  # Base reward
        completion_lower = completion.lower()
        
        for keyword in positive_keywords:
            if keyword in completion_lower:
                reward += 0.2
        
        for keyword in negative_keywords:
            if keyword in completion_lower:
                reward -= 0.2
        
        rewards.append(max(0, min(1, reward)))
    return rewards
```

#### 3. Custom Reward Model

```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained("reward-model")

def model_based_reward(completions, prompts):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        inputs = tokenizer(prompt + completion, return_tensors="pt")
        with torch.no_grad():
            outputs = reward_model(**inputs)
            reward = outputs.logits.item()
        rewards.append(reward)
    return rewards
```

### Reward Function Tips

1. **Normalization**: Scale rewards to a consistent range (e.g., [0, 1])
2. **Variance**: Ensure rewards have sufficient variance for learning
3. **Bias**: Avoid systematic bias in reward computation
4. **Interpretability**: Use interpretable rewards when possible
5. **Validation**: Test reward function on sample data

## Training Process

### Training Loop

```
For each epoch:
    For each batch:
        1. Sample prompts from dataset
        2. Generate multiple completions per prompt
        3. Compute rewards for each completion
        4. Calculate advantages using GAE
        5. Compute policy loss with clipping
        6. Add KL penalty
        7. Backpropagate and update
        8. Log metrics
    Save checkpoint
```

### Monitoring Training

Key metrics to monitor:

1. **Loss**: Should decrease over time
2. **Reward**: Should increase over time
3. **KL Divergence**: Should stay within bounds (0.05-0.1)
4. **Policy Ratio**: Should be close to 1.0
5. **Learning Rate**: Monitor for decay schedules

### Common Training Issues

#### Issue 1: Exploding Rewards

**Symptoms**: Rewards grow indefinitely
**Solutions**:
- Clip rewards to a maximum value
- Normalize rewards using running statistics
- Reduce learning rate

#### Issue 2: Mode Collapse

**Symptoms**: Model generates identical responses
**Solutions**:
- Increase temperature during generation
- Add entropy bonus
- Increase KL penalty

#### Issue 3: KL Divergence Too High

**Symptoms**: KL penalty term grows large
**Solutions**:
- Increase KL penalty coefficient
- Reduce learning rate
- Use adaptive KL penalty

## Optimization Strategies

### 1. Hyperparameter Tuning

#### Learning Rate
- Start with 1e-6
- If loss oscillates: decrease by 10x
- If learning is too slow: increase by 2x

#### Batch Size
- Start with 2 (24GB GPU)
- Adjust based on memory constraints
- Use gradient accumulation for effective batch size

#### KL Penalty
- Start with 0.1
- Increase if KL divergence grows
- Decrease if learning is too slow

### 2. Training Speed Optimization

#### Multi-GPU Training
```bash
accelerate launch --multi_gpu --num_processes=4 train_grpo.py
```

#### DeepSpeed
- Use ZeRO Stage 2 for memory efficiency
- Enable gradient checkpointing
- Use mixed precision (FP16)

#### Data Loading
```python
dataloader_kwargs = {
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
}
```

### 3. Memory Optimization

#### QLoRA Setup
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

#### 8-bit Quantization
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

## Troubleshooting Guide

### Common Errors and Solutions

#### Error 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `batch_size = 1`
2. Enable gradient checkpointing
3. Use quantization: `load_in_8bit = True`
4. Reduce sequence length: `max_length = 1024`
5. Use QLoRA instead of full fine-tuning

#### Error 2: Dataset Loading Error

**Symptoms**: `DatasetNotFoundError` or connection errors

**Solutions**:
1. Check internet connection
2. Verify dataset name in HuggingFace Hub
3. Use `datasets.load_dataset(..., download_mode="force_redownload")`
4. Try loading from local cache

#### Error 3: Model Loading Error

**Symptoms**: `OSError: Can't load config for ...`

**Solutions**:
1. Verify model name: `Qwen/Qwen2.5-3B-Instruct`
2. Check HuggingFace authentication: `huggingface-cli login`
3. Use `trust_remote_code=True`
4. Clear cache: `rm -rf ~/.cache/huggingface`

#### Error 4: Gradient Explosion

**Symptoms**: Loss becomes NaN or infinite

**Solutions**:
1. Reduce learning rate: `1e-7`
2. Enable gradient clipping: `max_grad_norm = 1.0`
3. Check reward function for extreme values
4. Use warmup: `warmup_ratio = 0.1`

### Debugging Tips

1. **Start Small**: Test with 100 examples first
2. **Verbose Logging**: Set `logging_steps = 1`
3. **Check Outputs**: Print generated responses
4. **Validate Rewards**: Compute and print reward statistics
5. **Monitor GPU**: Use `nvidia-smi -l 1` during training

### Performance Benchmarking

Expected training times (single A100 40GB):

| Configuration | Examples/Hour | Time/Epoch (53K) |
|--------------|---------------|------------------|
| FP16, batch=2 | ~500 | ~106 hours |
| FP16 + DeepSpeed | ~800 | ~66 hours |
| 8-bit + DeepSpeed | ~1200 | ~44 hours |
| QLoRA + DeepSpeed | ~2000 | ~26 hours |

Note: Actual performance depends on hardware and configuration.

## Best Practices

1. **Reproducibility**: Set random seeds (`torch.manual_seed(42)`)
2. **Checkpointing**: Save regularly (`save_steps = 500`)
3. **Monitoring**: Use WandB or TensorBoard
4. **Validation**: Evaluate on held-out set
5. **Documentation**: Record hyperparameters and results

## References

1. [TRL Documentation](https://huggingface.co/docs/trl)
2. [Qwen Models](https://huggingface.co/Qwen)
3. [PPO Paper](https://arxiv.org/abs/1707.06347)
4. [GAE Paper](https://arxiv.org/abs/1506.02438)

## Support

For issues or questions:
1. Check this guide first
2. Review TRL documentation
3. Open an issue on GitHub
4. Check HuggingFace forums

---

**Last Updated**: 2025-02-18
**Version**: 1.0