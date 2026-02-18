"""
Data Preprocessing Script for POLARIS Dataset

This script helps explore and preprocess the POLARIS dataset before training.
It's useful for understanding the dataset structure and preparing it for GRPO training.
"""

from datasets import load_dataset
import json
from collections import Counter
import matplotlib.pyplot as plt


def explore_dataset():
    """
    Explore the POLARIS dataset structure and statistics.
    """
    print("=" * 60)
    print("POLARIS Dataset Exploration")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
    
    print(f"\nDataset size: {len(dataset)} examples")
    print(f"\nDataset features: {dataset.features}")
    
    # Show first few examples
    print("\n" + "=" * 60)
    print("First 3 Examples:")
    print("=" * 60)
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i+1}:")
        example = dataset[i]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
    
    # Analyze text lengths
    print("\n" + "=" * 60)
    print("Text Length Statistics:")
    print("=" * 60)
    
    # Check which fields contain text
    text_fields = [k for k in dataset.features.keys() 
                  if dataset.features[k].dtype == 'string']
    
    for field in text_fields[:3]:  # Analyze first 3 text fields
        lengths = [len(example[field]) for example in dataset if field in example]
        if lengths:
            print(f"\n{field}:")
            print(f"  Min: {min(lengths)}")
            print(f"  Max: {max(lengths)}")
            print(f"  Mean: {sum(lengths)/len(lengths):.2f}")
            print(f"  Median: {sorted(lengths)[len(lengths)//2]:.2f}")
    
    # Save sample data to JSON for inspection
    print("\n" + "=" * 60)
    print("Saving sample data...")
    print("=" * 60)
    
    sample_data = []
    for i in range(min(100, len(dataset))):
        example = dataset[i]
        # Convert to JSON-serializable format
        serializable_example = {}
        for key, value in example.items():
            if hasattr(value, 'tolist'):  # Handle numpy arrays
                serializable_example[key] = value.tolist()
            else:
                serializable_example[key] = value
        sample_data.append(serializable_example)
    
    with open("sample_polaris_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved 100 examples to 'sample_polaris_data.json'")
    
    return dataset


def format_for_grpo(dataset, prompt_field="input", completion_field="output"):
    """
    Format dataset for GRPO training.
    
    Args:
        dataset: Raw dataset
        prompt_field: Field name containing the prompt/input
        completion_field: Field name containing the completion/output
        
    Returns:
        Formatted dataset
    """
    print("\n" + "=" * 60)
    print("Formatting Dataset for GRPO")
    print("=" * 60)
    
    def format_example(example):
        # Try different possible field names
        prompt = example.get(prompt_field) or example.get('question') or example.get('instruction') or ''
        completion = example.get(completion_field) or example.get('answer') or example.get('response') or ''
        
        # Format as conversation
        formatted_prompt = f"User: {prompt}\nAssistant: "
        
        return {
            "prompt": formatted_prompt,
            "completion": completion,
        }
    
    # Apply formatting
    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )
    
    print(f"\nFormatted {len(formatted_dataset)} examples")
    print("\nSample formatted example:")
    print(f"Prompt: {formatted_dataset[0]['prompt'][:200]}...")
    print(f"Completion: {formatted_dataset[0]['completion'][:200]}...")
    
    return formatted_dataset


def split_dataset(dataset, train_size=0.9, test_size=0.05, val_size=0.05, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_size: Fraction for training
        test_size: Fraction for testing
        val_size: Fraction for validation
        seed: Random seed
        
    Returns:
        DatasetDict with train, validation, test splits
    """
    print("\n" + "=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    
    # First split: train + val vs test
    train_val_test = dataset.train_test_split(
        test_size=test_size,
        seed=seed
    )
    
    # Second split: train vs val
    remaining = train_val_test["train"]
    val_fraction = val_size / (train_size + val_size)
    train_val = remaining.train_test_split(
        test_size=val_fraction,
        seed=seed
    )
    
    splits = {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_val_test["test"]
    }
    
    print(f"\nTrain set: {len(splits['train'])} examples")
    print(f"Validation set: {len(splits['validation'])} examples")
    print(f"Test set: {len(splits['test'])} examples")
    
    return splits


def save_formatted_dataset(formatted_dataset, output_path="formatted_polaris"):
    """
    Save formatted dataset to disk.
    
    Args:
        formatted_dataset: Dataset to save
        output_path: Output directory
    """
    print("\n" + "=" * 60)
    print("Saving Formatted Dataset")
    print("=" * 60)
    
    formatted_dataset.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")


def main():
    """Main preprocessing function."""
    
    # Step 1: Explore dataset
    dataset = explore_dataset()
    
    # Step 2: Format for GRPO
    # Note: Adjust field names based on actual dataset structure
    formatted_dataset = format_for_grpo(
        dataset,
        prompt_field="input",  # Change based on actual field name
        completion_field="output"  # Change based on actual field name
    )
    
    # Step 3: Split dataset
    splits = split_dataset(formatted_dataset)
    
    # Step 4: Save formatted dataset
    save_formatted_dataset(formatted_dataset)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review 'sample_polaris_data.json' to understand data format")
    print("2. Adjust field names in format_for_grpo() if needed")
    print("3. Use the formatted dataset in train_grpo.py")


if __name__ == "__main__":
    main()