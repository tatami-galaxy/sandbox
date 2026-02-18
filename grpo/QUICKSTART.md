# Quick Start Guide

Get started with GRPO training in 5 minutes!

## Prerequisites Check

- ✅ Python 3.9+
- ✅ GPU with 24GB+ VRAM (or use quantization for smaller GPUs)
- ✅ Internet connection

## Installation (2 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Preprocessing (1 minute)

```bash
python preprocess_data.py
```

This will:
- Load the POLARIS dataset
- Show dataset structure
- Save sample data for inspection

## Start Training (2 minutes)

### Option 1: Default Training (24GB GPU)

```bash
python train_grpo.py
```

### Option 2: With 8-bit Quantization (12GB GPU)

Edit `config.py`:
```python
"load_in_8bit": True
```

Then run:
```bash
python train_grpo.py
```

### Option 3: With QLoRA (8GB GPU)

Edit `config.py`:
```python
"use_lora": True,
"load_in_4bit": True,
```

Then run:
```bash
python train_grpo.py
```

## Monitor Training

**Terminal**: Watch progress in real-time

**TensorBoard** (in new terminal):
```bash
tensorboard --logdir=./logs/qwen-grpo-polaris
```

**WandB** (optional):
```bash
wandb login
```

## Test Trained Model

```bash
python inference.py --mode interactive
```

## What's Next?

1. **Adjust Configuration**: Edit `config.py` to change hyperparameters
2. **Customize Reward**: Modify `compute_reward_function` in `train_grpo.py`
3. **Read Documentation**: Check `README.md` for detailed instructions
4. **Advanced Topics**: See `IMPLEMENTATION_GUIDE.md` for deep dives

## Common First-Time Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Set `load_in_8bit = True` in `config.py` |
| Dataset not found | Check internet connection |
| Slow training | Use DeepSpeed or reduce `max_length` |
| High memory usage | Set `batch_size = 1` in `config.py` |

## Quick Configuration Reference

Edit these in `config.py`:

```python
# Reduce memory usage
"batch_size": 1,  # instead of 2
"max_length": 1024,  # instead of 2048

# Speed up training
"logging_steps": 50,  # instead of 10
"save_steps": 1000,  # instead of 500

# Use less data
"max_train_samples": 1000,  # instead of None (all data)
```

## Need Help?

- 📖 **README.md**: Full documentation
- 🔧 **IMPLEMENTATION_GUIDE.md**: Technical details
- 🐛 **Troubleshooting**: Check README.md "Troubleshooting" section

---

**Time to Train**: 5 minutes setup + several hours to train

Good luck! 🚀