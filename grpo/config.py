"""
Configuration file for GRPO training of Qwen on POLARIS dataset.

This file contains all hyperparameters and settings for training.
Modify these values to adjust the training configuration.
"""

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    # Model name from HuggingFace Hub
    # Options: "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", etc.
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    
    # Quantization settings (useful for memory efficiency)
    "load_in_8bit": False,  # Set to True for 8-bit quantization
    "load_in_4bit": False,  # Set to True for 4-bit quantization (requires bitsandbytes)
    
    # Precision
    "torch_dtype": "float16",  # Options: "float16", "bfloat16", "float32"
}


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_CONFIG = {
    # Dataset path
    "dataset_path": "POLARIS-Project/Polaris-Dataset-53K",
    
    # Field names in the dataset (adjust based on actual dataset structure)
    "prompt_field": "input",  # Options: "input", "question", "instruction"
    "completion_field": "output",  # Options: "output", "answer", "response"
    
    # Dataset split ratios
    "train_size": 0.95,
    "test_size": 0.05,
    "val_size": 0.0,  # Can add validation set if needed
    
    # Random seed for reproducibility
    "seed": 42,
    
    # Max samples to use (set to None to use all data)
    "max_train_samples": None,
    "max_eval_samples": None,
}


# =============================================================================
# Training Configuration
# =============================================================================
TRAINING_CONFIG = {
    # Basic training parameters
    "learning_rate": 1e-6,
    "batch_size": 2,  # Adjust based on GPU memory (2 for 24GB, 4 for 48GB, etc.)
    "gradient_accumulation_steps": 4,  # Effective batch size = batch_size * gradient_accumulation_steps
    "num_train_epochs": 3,
    
    # Sequence lengths
    "max_length": 2048,
    "max_prompt_length": 512,
    "max_completion_length": 1536,
    
    # Optimizer settings
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    
    # Memory optimization
    "gradient_checkpointing": True,
    "fp16": True,  # Use FP16 for training
    "bf16": False,  # Use BF16 if GPU supports it (A100, H100, etc.)
}


# =============================================================================
# GRPO-Specific Configuration
# =============================================================================
GRPO_CONFIG = {
    # PPO/GRPO clipping parameter
    "epsilon": 0.2,
    
    # Discount factor for rewards
    "gamma": 0.99,
    
    # GAE (Generalized Advantage Estimation) parameter
    "gae_lambda": 0.95,
    
    # KL divergence penalty coefficient
    "kl_penalty": 0.1,
    
    # Target KL divergence for early stopping
    "target_kl": 0.1,
    
    # Number of generations per prompt
    "num_generations": 4,
}


# =============================================================================
# Logging and Saving Configuration
# =============================================================================
LOGGING_CONFIG = {
    # Output directory
    "output_dir": "./output/qwen-grpo-polaris",
    
    # Logging directory
    "logging_dir": "./logs/qwen-grpo-polaris",
    
    # Logging frequency
    "logging_steps": 10,
    
    # Save frequency
    "save_steps": 500,
    "save_total_limit": 3,  # Keep only last 3 checkpoints
    
    # Evaluation frequency
    "eval_steps": 500,
    
    # Reporting tools
    "report_to": ["wandb", "tensorboard"],
    
    # WandB project name
    "wandb_project": "qwen-grpo-polaris",
    "wandb_entity": None,  # Set to your wandb username/team
    
    # Run name (None for auto-generated)
    "run_name": None,
}


# =============================================================================
# Hardware Configuration
# =============================================================================
HARDWARE_CONFIG = {
    # Device (auto-detected if None)
    "device": None,  # Options: "cuda", "cpu", None (auto)
    
    # Number of GPUs to use (for multi-GPU training)
    "num_gpus": 1,
    
    # Distributed training backend
    "distributed_backend": "nccl",  # Options: "nccl", "gloo"
    
    # DeepSpeed configuration (optional)
    "use_deepspeed": False,
    "deepspeed_config": "./deepspeed_config.json",
}


# =============================================================================
# Reward Function Configuration
# =============================================================================
REWARD_CONFIG = {
    # Reward function type
    "reward_function": "length_based",  # Options: "length_based", "keyword_match", "custom"
    
    # Length-based reward parameters
    "min_length": 10,
    "target_length": 100,
    "max_length": 1000,
    
    # Keyword-based reward parameters (if using keyword_match)
    "positive_keywords": ["correct", "yes", "true"],
    "negative_keywords": ["wrong", "no", "false", "error"],
    
    # Custom reward model (if using custom)
    "reward_model_path": None,  # Path to custom reward model
}


# =============================================================================
# Inference Configuration (for evaluation)
# =============================================================================
INFERENCE_CONFIG = {
    # Generation parameters
    "max_new_tokens": 512,
    "min_new_tokens": 10,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "num_return_sequences": 1,
    "do_sample": True,
}


# =============================================================================
# Advanced Configuration
# =============================================================================
ADVANCED_CONFIG = {
    # LoRA/QLoRA parameters (for parameter-efficient fine-tuning)
    "use_lora": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    
    # Early stopping
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
    
    # Data loading
    "dataloader_num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    
    # Miscellaneous
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": False,
}


# =============================================================================
# Helper Functions
# =============================================================================
def get_config():
    """
    Get all configuration as a dictionary.
    
    Returns:
        dict: Combined configuration
    """
    config = {}
    config.update(MODEL_CONFIG)
    config.update(DATASET_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(GRPO_CONFIG)
    config.update(LOGGING_CONFIG)
    config.update(HARDWARE_CONFIG)
    config.update(REWARD_CONFIG)
    config.update(INFERENCE_CONFIG)
    config.update(ADVANCED_CONFIG)
    
    return config


def update_config_from_args(args):
    """
    Update configuration from command-line arguments.
    
    Args:
        args: Namespace object from argparse
        
    Returns:
        dict: Updated configuration
    """
    config = get_config()
    
    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    
    return config


if __name__ == "__main__":
    # Print current configuration
    import json
    config = get_config()
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print(json.dumps(config, indent=2))