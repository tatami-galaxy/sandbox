"""
Simplified PPO Training Script for Qwen on POLARIS Dataset
Uses PPOTrainer directly without wrapper
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model


def main():
    """Main training function using PPOTrainer directly"""
    
    # Configuration
    model_name = "Qwen/Qwen2.5-4B"
    dataset_name = "POLARIS-Project/Polaris-Dataset-53K"
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        total_ppo_epochs=20,
        ppo_epochs=4,
        max_grad_norm=1.0,
        optimize_cuda_cache=True,
        target_kl=6.0,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
        log_with="wandb",  # Set to None to disable
        project_name="qwen-ppo-polaris",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA for parameter-efficient training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.get_nb_trainable_parameters()}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    # Limit dataset size for faster training (optional)
    dataset = dataset.select(range(min(len(dataset), ppo_config.batch_size * 100)))
    print(f"Using {len(dataset)} examples")
    
    # Data collator
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # PPOTrainer will create a reference model
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )
    
    # Generation arguments
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 256,
    }
    
    # Reward function
    def compute_rewards(prompts, outputs, **kwargs):
        """Simple heuristic reward function"""
        rewards = []
        for prompt, output in zip(prompts, outputs):
            reward = 0.0
            # Length reward
            output_length = len(output.split())
            if 10 <= output_length <= 500:
                reward += min(output_length * 0.01, 1.0)
            # Content reward
            if len(output.strip()) > 0:
                reward += 0.5
            # Diversity reward
            unique_words = set(output.lower().split())
            if len(unique_words) > 0:
                diversity_ratio = len(unique_words) / output_length
                reward += diversity_ratio * 0.5
            rewards.append(reward)
        return rewards
    
    # Training loop
    print("Starting PPO training...")
    
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        if epoch >= ppo_config.total_ppo_epochs:
            break
        
        print(f"\nEpoch {epoch + 1}/{ppo_config.total_ppo_epochs}")
        
        # Extract queries from batch
        query_tensors = []
        for example in batch:
            # Get prompt from appropriate field
            prompt = example.get("instruction", 
                   example.get("prompt", 
                   example.get("input", str(list(example.values())[0]))))
            
            query_tensor = tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).squeeze(0)
            query_tensors.append(query_tensor)
        
        # Pad query tensors
        query_tensors = torch.nn.utils.rnn.pad_sequence(
            [qt.to(model.device) for qt in query_tensors],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        
        # Generate responses
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        
        # Decode responses for reward computation
        batch["response"] = [
            tokenizer.decode(rt.squeeze(), skip_special_tokens=True)
            for rt in response_tensors
        ]
        
        # Get prompts for reward computation
        prompts = [
            b.get("instruction", 
             b.get("prompt", 
             b.get("input", str(list(b.values())[0]))))
            for b in batch
        ]
        
        # Compute rewards
        rewards = compute_rewards(prompts, batch["response"])
        rewards = [torch.tensor(r).to(model.device) for r in rewards]
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Print statistics
        print(f"Average reward: {torch.stack(rewards).mean().item():.4f}")
        print(f"PPO stats: {stats}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            output_dir = f"checkpoint_epoch_{epoch + 1}"
            print(f"Saving checkpoint to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    # Save final model
    print("\nSaving final model...")
    model.save_pretrained("qwen_ppo_polaris_final")
    tokenizer.save_pretrained("qwen_ppo_polaris_final")
    
    print("Training completed!")


if __name__ == "__main__":
    main()