"""
Inference Script for Trained Qwen Model

This script loads a trained Qwen model and performs inference on test examples.
Useful for evaluating the model after training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def load_model_and_tokenizer(model_path):
    """
    Load trained model and tokenizer.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print(f"Model loaded successfully")
    print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
):
    """
    Generate a response from the model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        str: Generated response
    """
    # Format prompt
    formatted_prompt = f"User: {prompt}\nAssistant: "
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048 - max_new_tokens,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response
    if "Assistant: " in full_text:
        response = full_text.split("Assistant: ")[-1].strip()
    else:
        response = full_text
    
    return response


def interactive_inference(model, tokenizer):
    """
    Run interactive inference mode.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
    """
    print("\n" + "=" * 60)
    print("Interactive Inference Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")
    
    generation_params = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
    }
    
    while True:
        # Get user input
        prompt = input("User: ").strip()
        
        # Check for quit command
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode...")
            break
        
        if not prompt:
            continue
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        response = generate_response(model, tokenizer, prompt, **generation_params)
        print(response)
        print()


def batch_inference(model, tokenizer, test_examples):
    """
    Run batch inference on test examples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_examples: List of test prompts
    """
    print("\n" + "=" * 60)
    print("Batch Inference Mode")
    print("=" * 60)
    
    generation_params = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
    }
    
    for i, example in enumerate(test_examples, 1):
        print(f"\nExample {i}:")
        print(f"Prompt: {example[:200]}..." if len(example) > 200 else f"Prompt: {example}")
        
        response = generate_response(model, tokenizer, example, **generation_params)
        print(f"Response: {response}")
        print("-" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inference for trained Qwen model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/qwen-grpo-polaris",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Inference mode: interactive or batch",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Set generation parameters
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 50,
        "do_sample": True,
    }
    
    # Run inference
    if args.mode == "interactive":
        interactive_inference(model, tokenizer)
    else:
        # Example test prompts
        test_examples = [
            "What is machine learning?",
            "Explain the concept of reinforcement learning.",
            "What is the difference between supervised and unsupervised learning?",
            "Describe the GRPO algorithm in simple terms.",
            "What are the key components of a transformer model?",
        ]
        batch_inference(model, tokenizer, test_examples)


if __name__ == "__main__":
    main()