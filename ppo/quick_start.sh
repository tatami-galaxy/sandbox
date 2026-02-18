#!/bin/bash

# Quick start script for PPO training
# This script helps you get started with training quickly

echo "=========================================="
echo "PPO Training Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Step 3: Choose configuration:"
echo "  1) Default (Qwen 1.5B, moderate resources)"
echo "  2) Low Memory (Qwen 1.5B, 4-bit quantization)"
echo "  3) High Quality (Qwen 4B, more epochs)"
echo ""
read -p "Enter your choice (1-3) [default: 1]: " choice

case $choice in
    2)
        echo "Using low memory configuration..."
        # Modify train_ppo.py to use low_memory config
        sed -i.bak 's/trainer = PPOTrainerWrapper(# Start with default/trainer = PPOTrainerWrapper(config=get_low_memory_config()/' train_ppo.py
        ;;
    3)
        echo "Using high quality configuration..."
        # Modify train_ppo.py to use high_quality config
        sed -i.bak 's/trainer = PPOTrainerWrapper(# Start with default/trainer = PPOTrainerWrapper(config=get_high_quality_config()/' train_ppo.py
        ;;
    *)
        echo "Using default configuration..."
        ;;
esac

echo ""
echo "Step 4: Starting training..."
echo "Note: This may take several hours depending on your GPU"
echo ""

# Ask if user wants to use Weights & Biases
read -p "Do you want to log to Weights & Biases? (y/n) [default: n]: " use_wandb

if [ "$use_wandb" = "y" ] || [ "$use_wandb" = "Y" ]; then
    if command -v wandb &> /dev/null; then
        wandb login
    else
        echo "Warning: wandb not found. Install with: pip install wandb"
    fi
fi

# Run training
python train_ppo.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Your model is saved in: qwen_ppo_polaris_final/"
echo ""
echo "To use your trained model, run:"
echo "  python inference_example.py"
echo ""
echo "For interactive mode:"
echo "  python inference_example.py --interactive"
echo ""