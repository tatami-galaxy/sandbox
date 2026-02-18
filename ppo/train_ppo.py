"""
PPO Training Script for Qwen 4B on POLARIS Dataset

This script trains a Qwen 4B model using the Proximal Policy Optimization (PPO)
algorithm from TRL library on the POLARIS-Project/Polaris-Dataset-53K dataset.
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
import wandb
import os


def compute_reward(response_text):
    """
    Simple reward function for PPO training.
    
    Args:
        response_text: Generated response
        
    Returns:
        float: Reward value
    """
    # Simple reward based on length (encourage meaningful responses)
    # Normalize to [0, 1] roughly
    return min(len(response_text.split()) / 100.0, 1.0)


def build_dataset(config, dataset_name="POLARIS-Project/Polaris-Dataset-53K"):
    """
    Load and preprocess the dataset for PPO.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    ds = load_dataset(dataset_name, split="train")
    
    def tokenize(sample):
        # Format prompt
        prompt = f"User: {sample.get('input', sample.get('question', ''))}\nAssistant: "
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = prompt
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds, tokenizer


def collator(data):
    """
    Custom collator to handle dictionary of lists.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])


def main():
    # Configuration
    config = PPOConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        learning_rate=1.41e-5,
        batch_size=1,  # Small batch size for memory efficiency
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
        log_with="wandb",
        tracker_project_name="qwen-ppo-polaris",
    )

    # Initialize wandb
    wandb.init(project="qwen-ppo-polaris")

    # Load dataset and tokenizer
    dataset, tokenizer = build_dataset(config)

    # Load model with value head
    # Note: Qwen models require trust_remote_code=True
    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        # load_in_8bit=True,  # Uncomment if memory is tight
    )

    # Create reference model
    print("Creating reference model...")
    ref_model = create_reference_model(model)

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # Generation settings
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 256,
    }

    # Training loop
    print("Starting PPO training...")
    output_dir = "./output/qwen-ppo-polaris"
    os.makedirs(output_dir, exist_ok=True)
    
    save_steps = 500
    total_steps = 0

    # Iterate over the dataset
    # Note: In a real scenario, you might want to limit the number of steps
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # 1. Generate response
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )
        
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # 2. Compute rewards
        rewards = [torch.tensor(compute_reward(r)) for r in batch["response"]]

        # 3. Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log stats
        ppo_trainer.log_stats(stats, batch, rewards)
        
        total_steps += 1
        
        # Save model periodically
        if total_steps % save_steps == 0:
            print(f"Saving model at step {total_steps}...")
            ppo_trainer.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print("Training completed!")
    ppo_trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()