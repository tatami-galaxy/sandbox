"""
GRPO Training Script for Qwen 4B on POLARIS Dataset

This script trains a Qwen 4B model using the Group Relative Policy Optimization (GRPO)
algorithm from TRL library on the POLARIS-Project/Polaris-Dataset-53K dataset.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import GRPOTrainer, GRPOConfig
import wandb


def load_polaris_dataset():
    """
    Load and preprocess the POLARIS dataset.
    
    Returns:
        Dataset: Preprocessed dataset ready for training
    """
    print("Loading POLARIS dataset...")
    dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    print(f"First example:\n{dataset[0]}")
    
    return dataset


def prepare_dataset_for_grpo(dataset, tokenizer, max_length=2048):
    """
    Prepare dataset for GRPO training.
    
    Args:
        dataset: Raw dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Dataset: Prepared dataset with prompt and completion fields
    """
    def preprocess_function(examples):
        # Assuming POLARIS dataset has 'input' and 'output' fields
        # Adjust based on actual dataset structure
        prompts = []
        completions = []
        
        for example in examples:
            # Format as instruction-following task
            prompt = f"User: {example.get('input', example.get('question', ''))}\nAssistant: "
            completion = example.get('output', example.get('answer', ''))
            
            prompts.append(prompt)
            completions.append(completion)
        
        return {
            "prompt": prompts,
            "completion": completions,
        }
    
    # Apply preprocessing
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x),
        batched=False,
        remove_columns=dataset.column_names,
    )
    
    return processed_dataset


def create_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct", load_in_8bit=False):
    """
    Create and initialize model and tokenizer.
    
    Args:
        model_name: Model name from HuggingFace hub
        load_in_8bit: Whether to load model in 8-bit quantization
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def compute_reward_function(completions, **kwargs):
    """
    Simple reward function for GRPO training.
    This can be customized based on your specific task.
    
    Args:
        completions: Generated completions
        **kwargs: Additional arguments
        
    Returns:
        list: Reward values for each completion
    """
    # Simple reward based on length (can be replaced with more sophisticated metrics)
    rewards = []
    for completion in completions:
        # Reward based on completion length (encourage meaningful responses)
        reward = min(len(completion.split()) / 100.0, 1.0)
        rewards.append(reward)
    
    return rewards


def main():
    """Main training function."""
    
    # Initialize wandb (optional)
    wandb.init(
        project="qwen-grpo-polaris",
        config={
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dataset": "POLARIS-Project/Polaris-Dataset-53K",
            "algorithm": "GRPO",
        }
    )
    
    # Configuration
    config = GRPOConfig(
        # Model configuration
        model_name="Qwen/Qwen2.5-3B-Instruct",
        
        # Training hyperparameters
        learning_rate=1e-6,
        batch_size=2,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        max_length=2048,
        max_prompt_length=512,
        max_completion_length=1536,
        
        # GRPO-specific parameters
        epsilon=0.2,  # PPO clipping parameter
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        kl_penalty=0.1,  # KL divergence penalty
        
        # Optimizer settings
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Logging and saving
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        output_dir="./output/qwen-grpo-polaris",
        logging_dir="./logs/qwen-grpo-polaris",
        
        # Precision and device
        fp16=True,
        
        # Other settings
        report_to=["wandb", "tensorboard"],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # Load dataset
    dataset = load_polaris_dataset()
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_8bit=False,  # Set to True for memory efficiency
    )
    
    # Prepare dataset for GRPO
    processed_dataset = prepare_dataset_for_grpo(dataset, tokenizer)
    
    # Split dataset
    train_test_split = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_function=compute_reward_function,
    )
    
    # Train model
    print("\nStarting GRPO training...")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {config.output_dir}")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()