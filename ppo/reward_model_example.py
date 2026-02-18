"""
Example reward model implementation for PPO training
This file demonstrates how to implement different reward functions
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import numpy as np


class RewardModel:
    """Base class for reward models"""
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for generated outputs
        
        Args:
            prompts: Input prompts
            outputs: Generated outputs
            **kwargs: Additional arguments
            
        Returns:
            List of reward values
        """
        raise NotImplementedError


class HeuristicRewardModel(RewardModel):
    """Simple heuristic-based reward function"""
    
    def __init__(
        self,
        length_weight: float = 0.01,
        length_max_reward: float = 1.0,
        min_length: int = 10,
        max_length: int = 500,
        content_reward: float = 0.5
    ):
        """
        Initialize heuristic reward model
        
        Args:
            length_weight: Weight for length-based reward
            length_max_reward: Maximum reward for length
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            content_reward: Base reward for having content
        """
        self.length_weight = length_weight
        self.length_max_reward = length_max_reward
        self.min_length = min_length
        self.max_length = max_length
        self.content_reward = content_reward
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """Compute heuristic rewards"""
        rewards = []
        
        for prompt, output in zip(prompts, outputs):
            reward = 0.0
            
            # Length-based reward
            output_length = len(output.split())
            if self.min_length <= output_length <= self.max_length:
                reward += min(output_length * self.length_weight, self.length_max_reward)
            
            # Content reward
            if len(output.strip()) > 0:
                reward += self.content_reward
            
            # Diversity reward (avoid repetition)
            unique_words = set(output.lower().split())
            if len(unique_words) > 0:
                diversity_ratio = len(unique_words) / output_length
                reward += diversity_ratio * 0.5
            
            rewards.append(reward)
        
        return rewards


class TransformerRewardModel(RewardModel):
    """Reward model based on a trained transformer classifier"""
    
    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large",
        device: str = "cuda",
        batch_size: int = 4
    ):
        """
        Initialize transformer reward model
        
        Args:
            model_name: Hugging Face model ID
            device: Device to run on
            batch_size: Batch size for inference
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        print(f"Loading reward model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print("Reward model loaded successfully")
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """Compute rewards using transformer model"""
        rewards = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_outputs = outputs[i:i + self.batch_size]
            
            # Format inputs as [CLS] prompt [SEP] response [SEP]
            texts = [
                f"{prompt} {output}"
                for prompt, output in zip(batch_prompts, batch_outputs)
            ]
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute rewards
            with torch.no_grad():
                outputs_logits = self.model(**inputs).logits
                batch_rewards = outputs_logits.squeeze(-1).cpu().tolist()
            
            # Normalize rewards to [0, 1] range
            batch_rewards = [
                (r + 1) / 2  # Assuming logits are in [-1, 1] range
                for r in batch_rewards
            ]
            
            rewards.extend(batch_rewards)
        
        return rewards


class HybridRewardModel(RewardModel):
    """Hybrid reward model combining heuristic and model-based rewards"""
    
    def __init__(
        self,
        heuristic_model: Optional[RewardModel] = None,
        transformer_model: Optional[RewardModel] = None,
        heuristic_weight: float = 0.3,
        model_weight: float = 0.7
    ):
        """
        Initialize hybrid reward model
        
        Args:
            heuristic_model: Heuristic reward model
            transformer_model: Transformer reward model
            heuristic_weight: Weight for heuristic rewards
            model_weight: Weight for model rewards
        """
        self.heuristic_model = heuristic_model or HeuristicRewardModel()
        self.transformer_model = transformer_model
        self.heuristic_weight = heuristic_weight
        self.model_weight = model_weight
        
        # Normalize weights
        total_weight = heuristic_weight + model_weight
        self.heuristic_weight /= total_weight
        self.model_weight /= total_weight
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """Compute hybrid rewards"""
        # Get heuristic rewards
        heuristic_rewards = self.heuristic_model.compute_reward(prompts, outputs, **kwargs)
        
        # Get model rewards if available
        if self.transformer_model is not None:
            model_rewards = self.transformer_model.compute_reward(prompts, outputs, **kwargs)
        else:
            model_rewards = [0.0] * len(prompts)
        
        # Combine rewards
        combined_rewards = [
            self.heuristic_weight * hr + self.model_weight * mr
            for hr, mr in zip(heuristic_rewards, model_rewards)
        ]
        
        return combined_rewards


class CoherenceRewardModel(RewardModel):
    """Reward model based on text coherence and quality metrics"""
    
    def __init__(self):
        """Initialize coherence reward model"""
        from rouge_score import rouge_scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """Compute coherence-based rewards"""
        rewards = []
        
        for prompt, output in zip(prompts, outputs):
            reward = 0.0
            
            # Length reward
            output_words = output.split()
            if 20 <= len(output_words) <= 300:
                reward += 0.5
            
            # ROUGE score (comparing with expected format if available)
            if "reference" in kwargs and kwargs["reference"]:
                reference = kwargs["reference"][0] if isinstance(kwargs["reference"], list) else kwargs["reference"]
                scores = self.rouge_scorer.score(reference, output)
                rouge_score = scores['rougeL'].fmeasure
                reward += rouge_score * 0.5
            
            # Sentence structure reward
            sentences = output.split('.')
            if len(sentences) >= 2 and len(sentences) <= 10:
                reward += 0.3
            
            # Vocabulary diversity
            unique_words = set(word.lower() for word in output_words)
            if len(output_words) > 0:
                diversity = len(unique_words) / len(output_words)
                reward += diversity * 0.2
            
            rewards.append(min(reward, 1.0))
        
        return rewards


class CustomRewardModel(RewardModel):
    """Custom reward model - implement your own logic here"""
    
    def compute_reward(self, prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        """
        Implement your custom reward function here
        
        Example ideas:
        - Check for specific keywords or phrases
        - Use external APIs for quality assessment
        - Implement domain-specific logic
        - Use additional metadata from kwargs
        """
        rewards = []
        
        for prompt, output in zip(prompts, outputs):
            reward = 0.0
            
            # Your custom logic here
            # Example: reward for containing specific keywords
            keywords = ["answer", "explanation", "because", "therefore"]
            keyword_count = sum(1 for word in output.lower().split() if word in keywords)
            reward += keyword_count * 0.1
            
            # Example: penalize very short responses
            if len(output.split()) < 10:
                reward -= 0.3
            
            rewards.append(reward)
        
        return rewards


def get_reward_model(reward_type: str, **kwargs) -> RewardModel:
    """
    Factory function to get reward model
    
    Args:
        reward_type: Type of reward model ("heuristic", "model", "hybrid", "coherence", "custom")
        **kwargs: Additional arguments for the reward model
        
    Returns:
        RewardModel instance
    """
    if reward_type == "heuristic":
        return HeuristicRewardModel(**kwargs)
    elif reward_type == "model":
        return TransformerRewardModel(**kwargs)
    elif reward_type == "hybrid":
        heuristic = kwargs.pop("heuristic_model", HeuristicRewardModel())
        transformer = kwargs.pop("transformer_model", None)
        return HybridRewardModel(
            heuristic_model=heuristic,
            transformer_model=transformer,
            **kwargs
        )
    elif reward_type == "coherence":
        return CoherenceRewardModel(**kwargs)
    elif reward_type == "custom":
        return CustomRewardModel(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# Example usage
if __name__ == "__main__":
    # Test different reward models
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    outputs = [
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously."
    ]
    
    print("Testing Heuristic Reward Model:")
    heuristic_model = HeuristicRewardModel()
    rewards = heuristic_model.compute_reward(prompts, outputs)
    for p, o, r in zip(prompts, outputs, rewards):
        print(f"  Reward: {r:.4f} | Prompt: {p[:50]}... | Output: {o[:50]}...")
    
    print("\nTesting Coherence Reward Model:")
    coherence_model = CoherenceRewardModel()
    rewards = coherence_model.compute_reward(prompts, outputs)
    for p, o, r in zip(prompts, outputs, rewards):
        print(f"  Reward: {r:.4f} | Prompt: {p[:50]}... | Output: {o[:50]}...")
    
    print("\nTesting Custom Reward Model:")
    custom_model = CustomRewardModel()
    rewards = custom_model.compute_reward(prompts, outputs)
    for p, o, r in zip(prompts, outputs, rewards):
        print(f"  Reward: {r:.4f} | Prompt: {p[:50]}... | Output: {o[:50]}...")