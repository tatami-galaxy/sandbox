"""
PPO Training Script for Qwen 4B on POLARIS-Dataset-53K
Uses TRL library from Hugging Face
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from typing import List, Dict
import numpy as np


class PPOTrainerWrapper:
    """Wrapper for PPO training with Qwen 4B model"""
    
    def __init__(self, config: PPOConfig, model_name: str = "Qwen/Qwen2.5-1.5B"):
        """
        Initialize PPO trainer
        
        Args:
            config: PPO configuration
            model_name: Model identifier from Hugging Face Hub
        """
        self.config = config
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Apply LoRA for parameter-efficient training
        self._apply_lora()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize PPO trainer
        self.ppo_trainer = None
        
    def _apply_lora(self):
        """Apply LoRA adapters for efficient training"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"Trainable parameters: {self.model.get_nb_trainable_parameters()}")
    
    def _load_dataset(self):
        """Load and preprocess POLARIS dataset"""
        print("Loading POLARIS-Project/Polaris-Dataset-53K dataset...")
        dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        
        # Filter to a smaller subset for faster training if needed
        if len(dataset) > self.config.total_ppo_epochs * self.config.batch_size:
            dataset = dataset.select(range(min(len(dataset), self.config.batch_size * 100)))
        
        print(f"Loaded {len(dataset)} examples from dataset")
        return dataset
    
    def _collator(self, data: List[Dict]):
        """Collate function for batch processing"""
        return {key: [d[key] for d in data] for key in data[0]}
    
    def _compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """
        Compute reward for generated outputs
        This is a simple reward function - replace with your own or load a reward model
        
        Args:
            prompts: Input prompts
            outputs: Generated outputs
            
        Returns:
            List of reward values
        """
        rewards = []
        for prompt, output in zip(prompts, outputs):
            # Simple heuristic rewards (replace with actual reward model)
            reward = 0.0
            
            # Reward based on output length
            reward += min(len(output.split()) * 0.01, 1.0)
            
            # Reward for having some content
            if len(output.strip()) > 0:
                reward += 0.5
            
            # Reward for reasonable length
            if 10 <= len(output.split()) <= 500:
                reward += 0.5
            
            rewards.append(reward)
        
        return rewards
    
    def generate_query_responses(self, batch: Dict) -> Dict[str, List]:
        """
        Generate queries and responses from batch
        
        Args:
            batch: Batch of examples
            
        Returns:
            Dictionary with queries and responses
        """
        queries = []
        responses = []
        
        for example in batch:
            # Adapt based on actual dataset structure
            if "instruction" in example:
                query = example["instruction"]
            elif "prompt" in example:
                query = example["prompt"]
            elif "input" in example:
                query = example["input"]
            else:
                # Fallback: use first text field
                for key, value in example.items():
                    if isinstance(value, str):
                        query = value
                        break
                else:
                    query = "Please provide a helpful response."
            
            queries.append(query)
        
        # Generate responses using the model
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                min_length=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode responses
        for i, output in enumerate(outputs):
            # Remove the input prompt from the output
            response = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)
        
        return {"query": queries, "response": responses}
    
    def train(self):
        """Main training loop"""
        print("Initializing PPO trainer...")
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.config,
            model=self.model,
            ref_model=None,  # Will use the same model as reference
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            data_collator=self._collator
        )
        
        print("Starting PPO training...")
        
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": 256,
        }
        
        # Training loop
        for epoch, batch in enumerate(self.ppo_trainer.dataloader):
            if epoch >= self.config.total_ppo_epochs:
                break
            
            query_tensors = []
            response_tensors = []
            
            # Prepare batch
            print(f"\nEpoch {epoch + 1}/{self.config.total_ppo_epochs}")
            
            # Tokenize queries
            for example in batch:
                if "instruction" in example:
                    query = example["instruction"]
                elif "prompt" in example:
                    query = example["prompt"]
                elif "input" in example:
                    query = example["input"]
                else:
                    query = str(list(example.values())[0])
                
                query_tensor = self.tokenizer.encode(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).squeeze(0)
                query_tensors.append(query_tensor)
            
            # Pad query tensors to same length
            query_tensors = torch.nn.utils.rnn.pad_sequence(
                [qt.to(self.device) for qt in query_tensors],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            
            # Generate responses
            response_tensors = self.ppo_trainer.generate(
                query_tensors,
                **generation_kwargs
            )
            
            # Decode responses for reward computation
            batch["response"] = [
                self.tokenizer.decode(rt.squeeze(), skip_special_tokens=True)
                for rt in response_tensors
            ]
            
            # Compute rewards
            rewards = self._compute_reward(
                prompts=[str(b.get("instruction", b.get("prompt", ""))) for b in batch],
                outputs=batch["response"]
            )
            
            # Convert to tensors
            rewards = [torch.tensor(r).to(self.device) for r in rewards]
            
            # Run PPO step
            stats = self.ppo_trainer.step(
                query_tensors,
                response_tensors,
                rewards
            )
            
            # Print statistics
            print(f"Average reward: {torch.stack(rewards).mean().item():.4f}")
            print(f"PPO stats: {stats}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
    
    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint"""
        print(f"Saving checkpoint to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Checkpoint saved!")


def main():
    """Main function to run PPO training"""
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name="Qwen/Qwen2.5-1.5B",  # Using smaller model for efficiency
        learning_rate=1.41e-5,
        batch_size=4,  # Small batch size for memory efficiency
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        total_ppo_epochs=20,
        ppo_epochs=4,
        max_grad_norm=1.0,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=6.0,
        init_kl_coef=0.2,
        adap_kl_ctrl=True,
        log_with="wandb",  # Set to None to disable wandb
        project_name="qwen-ppo-polaris",
        tracker_kwargs={
            "wandb": {"entity": "your-entity", "project": "qwen-ppo-polaris"}
        }
    )
    
    # Initialize trainer
    trainer = PPOTrainerWrapper(
        config=ppo_config,
        model_name="Qwen/Qwen2.5-1.5B"  # Or "Qwen/Qwen2.5-4B" if you have enough memory
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_checkpoint("qwen_ppo_polaris_final")
    
    print("Training completed!")


if __name__ == "__main__":
    main()