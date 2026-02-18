"""
Configuration file for PPO training
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the model"""
    model_name: str = "Qwen/Qwen2.5-4B"  # Change to "Qwen/Qwen2.5-1.5B" for smaller model
    use_4bit: bool = False  # Use 4-bit quantization for memory efficiency
    use_flash_attention: bool = True
    torch_dtype: str = "float16"  # Options: "float32", "float16", "bfloat16"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class DatasetConfig:
    """Configuration for the dataset"""
    dataset_name: str = "POLARIS-Project/Polaris-Dataset-53K"
    dataset_split: str = "train"
    max_examples: Optional[int] = None  # Set to limit dataset size, None for all
    max_length: int = 512
    prompt_field: str = "instruction"  # Field to use as prompt
    
    # Dataset field mappings (may need adjustment based on actual dataset structure)
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training"""
    # Training parameters
    learning_rate: float = 1.41e-5
    batch_size: int = 4  # Reduce if OOM
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    total_ppo_epochs: int = 20
    ppo_epochs: int = 4  # PPO epochs per batch
    max_grad_norm: float = 1.0
    
    # PPO-specific parameters
    target_kl: float = 6.0
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True
    early_stopping: bool = False
    
    # Generation parameters
    max_new_tokens: int = 256
    min_new_tokens: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: float = 0.0
    do_sample: bool = True
    
    # Optimization
    optimize_cuda_cache: bool = True
    
    # Logging
    log_with: str = "wandb"  # Options: "wandb", "tensorboard", None
    project_name: str = "qwen-ppo-polaris"
    run_name: Optional[str] = None  # Auto-generated if None
    wandb_entity: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 5  # Save checkpoint every N epochs


@dataclass
class RewardConfig:
    """Configuration for reward computation"""
    reward_type: str = "heuristic"  # Options: "heuristic", "model", "hybrid"
    
    # For heuristic rewards
    length_weight: float = 0.01
    length_max_reward: float = 1.0
    min_length: int = 10
    max_length: int = 500
    content_reward: float = 0.5
    
    # For model-based rewards
    reward_model_name: Optional[str] = None  # e.g., "OpenAssistant/reward-model-deberta-v3-large"
    reward_batch_size: int = 4
    
    # For hybrid rewards
    heuristic_weight: float = 0.3
    model_weight: float = 0.7


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing"""
    output_dir: str = "./checkpoints"
    checkpoint_dir: str = "./checkpoints/intermediate"
    final_model_dir: str = "./qwen_ppo_polaris_final"
    
    save_total_limit: int = 3
    save_strategy: str = "epoch"  # Options: "epoch", "steps"
    
    # Push to hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


@dataclass
class TrainingConfig:
    """Main configuration class combining all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    ppo: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device configuration
    device: Optional[str] = None  # Auto-detect if None
    num_workers: int = 4


def get_default_config() -> TrainingConfig:
    """Get default training configuration"""
    return TrainingConfig()


def get_low_memory_config() -> TrainingConfig:
    """Get configuration for low-memory environments"""
    config = TrainingConfig()
    config.model.model_name = "Qwen/Qwen2.5-1.5B"
    config.model.use_4bit = True
    config.ppo.batch_size = 2
    config.ppo.mini_batch_size = 2
    config.ppo.max_new_tokens = 128
    return config


def get_high_quality_config() -> TrainingConfig:
    """Get configuration for higher quality training (requires more resources)"""
    config = TrainingConfig()
    config.ppo.batch_size = 8
    config.ppo.mini_batch_size = 8
    config.ppo.total_ppo_epochs = 50
    config.ppo.max_new_tokens = 512
    config.reward.reward_type = "model"
    config.reward.reward_model_name = "OpenAssistant/reward-model-deberta-v3-large"
    return config


# Configuration presets
CONFIG_PRESETS = {
    "default": get_default_config,
    "low_memory": get_low_memory_config,
    "high_quality": get_high_quality_config,
}