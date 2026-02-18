"""
Example script for inference with a PPO-trained Qwen model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def load_trained_model(
    base_model_name: str = "Qwen/Qwen2.5-4B",
    lora_path: str = "qwen_ppo_polaris_final",
    device: str = "auto"
):
    """
    Load the PPO-trained model with LoRA adapters
    
    Args:
        base_model_name: Name of the base model
        lora_path: Path to the saved LoRA adapters
        device: Device to load model on
        
    Returns:
        Model and tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {lora_path}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1
):
    """
    Generate a response from the model
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        repetition_penalty: Repetition penalty
        
    Returns:
        Generated response text
    """
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
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
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the output
    response = generated_text[len(prompt):].strip()
    
    return response


def batch_generate(
    model,
    tokenizer,
    prompts: list,
    **generation_kwargs
):
    """
    Generate responses for multiple prompts
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        prompts: List of input prompts
        **generation_kwargs: Generation parameters
        
    Returns:
        List of generated responses
    """
    responses = []
    
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt, **generation_kwargs)
        responses.append(response)
    
    return responses


def interactive_mode(model, tokenizer):
    """
    Run interactive chat mode
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
    """
    print("\n" + "="*50)
    print("Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        prompt = input("You: ")
        
        # Check for exit
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt.strip():
            continue
        
        # Generate response
        print("Assistant: ", end="", flush=True)
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser(description="Inference with PPO-trained Qwen model")
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-4B",
        help="Base model name or path"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default="qwen_ppo_polaris_final",
        help="Path to LoRA adapters"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate response for"
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save responses"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_trained_model(
        base_model_name=args.base_model,
        lora_path=args.lora_path
    )
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer)
        return
    
    # Single prompt
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        response = generate_response(model, tokenizer, args.prompt, **generation_kwargs)
        print(f"Response: {response}\n")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Response: {response}\n")
            print(f"Response saved to {args.output_file}")
        
        return
    
    # Batch prompts from file
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(prompts)} prompts...")
        responses = batch_generate(model, tokenizer, prompts, **generation_kwargs)
        
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"\n{'='*60}")
            print(f"Prompt {i}: {prompt}")
            print(f"Response {i}: {response}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                    f.write(f"{'='*60}\n")
                    f.write(f"Prompt {i}: {prompt}\n")
                    f.write(f"Response {i}: {response}\n\n")
            print(f"\nAll responses saved to {args.output_file}")
        
        return
    
    # Default: show example prompts
    example_prompts = [
        "What is machine learning?",
        "Explain the concept of reinforcement learning.",
        "Write a short poem about technology.",
        "What are the benefits of using PPO for language model training?"
    ]
    
    print("\nRunning example prompts...\n")
    for prompt in example_prompts:
        print(f"Prompt: {prompt}")
        response = generate_response(model, tokenizer, prompt, **generation_kwargs)
        print(f"Response: {response}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()