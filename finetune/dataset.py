"""
Dataset module for loading and processing vision-language datasets.
"""

from datasets import load_dataset
from typing import List, Dict, Any, Tuple
import PIL.Image
import os
import tempfile

def convert_to_conversation(sample: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert a raw VQA sample to a conversation format compatible with fine-tuning.
    
    Args:
        sample: Raw sample with question, choices, and image
        
    Returns:
        Formatted conversation for model training
    """
    question = sample["question"]
    choices = sample["choices"]
    correct_answer = sample["correct_answer"]
    
    # Map the answer letter (A, B, C, D) to index (0, 1, 2, 3)
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    correct_idx = letter_to_idx.get(correct_answer, 0)
    
    # Create a formatted question with choices
    formatted_question = f"{question}\nChoices:\n"
    for i, choice in enumerate(choices):
        formatted_question += f"{chr(65 + i)}. {choice}\n"
    
    # Create the answer using the correct choice
    answer = f"{correct_answer}. {choices[correct_idx]}"
    
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": formatted_question},
             {"type": "image", "image": sample["image"]}
         ]
        },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": answer}
         ]
        },
    ]
    return {"messages": conversation}

def load_training_dataset(config):
    """
    Load and prepare dataset for training.
    
    Args:
        config: Configuration object with dataset parameters
        
    Returns:
        Tuple of (train_split, val_split)
    """
    # Load dataset
    dataset = load_dataset(
        config.training.dataset_name, 
        split=config.training.dataset_split
    )
    
    # Limit the number of training samples if specified
    if config.training.max_train_samples is not None:
        if config.training.max_train_samples < len(dataset):
            dataset = dataset.select(range(config.training.max_train_samples))
            print(f"Limited dataset to {config.training.max_train_samples} samples for training")
    
    # Convert each sample to conversation format
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    
    from sklearn.model_selection import train_test_split
    # Split into train and validation
    train_split, val_split = train_test_split(
        converted_dataset, 
        test_size=config.training.validation_split, 
        random_state=config.model.random_state
    )
    
    # Log dataset info
    print(f"Dataset: {config.training.dataset_name}")
    print(f"Total samples: {len(converted_dataset)}")
    print(f"Training samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    
    return train_split, val_split

def load_evaluation_dataset(config):
    """
    Load dataset for evaluation.
    
    Args:
        config: Configuration with evaluation parameters
        
    Returns:
        Tuple of (test_cases, all_tags)
    """
    try:
        # Try to load the test split
        try:
            dataset = load_dataset(
                config.training.dataset_name, 
                split=config.evaluation.test_split
            )
            print(f"Successfully loaded {len(dataset)} samples from test split")
        except ValueError as e:
            if config.evaluation.fallback_to_train:
                # If test split doesn't exist, use a portion of train split
                print("Test split not found. Using a portion of train split instead.")
                dataset = load_dataset(
                    config.training.dataset_name, 
                    split=config.training.dataset_split
                )
                # Use last 20% of train data as test data
                train_size = len(dataset)
                test_size = int(train_size * 0.2)
                test_indices = list(range(train_size - test_size, train_size))
                dataset = dataset.select(test_indices)
                print(f"Selected {len(dataset)} samples from end of train split")
            else:
                raise e
        
        # Limit the number of samples if needed
        if config.evaluation.max_test_samples < len(dataset):
            dataset = dataset.select(range(config.evaluation.max_test_samples))
            print(f"Limited dataset to {config.evaluation.max_test_samples} samples for evaluation")
        
        # Create test cases
        test_cases = []
        all_tags = set()
        temp_dir = tempfile.mkdtemp()
        
        print(f"Processing test cases and saving images to temporary directory: {temp_dir}")
        
        for i, item in enumerate(dataset):
            try:
                # Extract question text
                question_text = item["question"]
                
                # Extract choices, handling both string and dict formats
                choices = item["choices"]
                formatted_question = f"{question_text}\nChoices:\n"
                
                # Check if choices are dictionaries or strings
                if isinstance(choices[0], dict) and "text" in choices[0]:
                    # Format choices as A, B, C, D (dict format with 'text' key)
                    for j, choice in enumerate(choices):
                        letter = chr(65 + j)  # A, B, C, D
                        formatted_question += f"{letter}. {choice['text']}\n"
                else:
                    # Format choices as A, B, C, D (string format)
                    for j, choice in enumerate(choices):
                        letter = chr(65 + j)  # A, B, C, D
                        formatted_question += f"{letter}. {choice}\n"
                
                # Get correct answer
                correct_answer = item["correct_answer"]
                
                # Handle image - could be PIL image or path
                if isinstance(item["image"], PIL.Image.Image):
                    # Save PIL image to a temporary file
                    img = item["image"]
                    img_path = os.path.join(temp_dir, f"image_{i}.jpg")
                    img.save(img_path)
                else:
                    # Use the image path directly
                    img_path = item["image"]
                    # If it's a dict with filename, use that
                    if isinstance(img_path, dict) and "filename" in img_path:
                        img_path = img_path["filename"]
                
                # Extract tag from metadata or ID
                tag = "general"
                # Try to get from metadata if available
                if "metadata" in item and isinstance(item["metadata"], dict) and "tag" in item["metadata"]:
                    tag = item["metadata"]["tag"]
                # Otherwise extract from ID
                elif "id" in item and item["id"]:
                    parts = item["id"].split("_")
                    tag = parts[0] if parts else "general"
                
                all_tags.add(tag)
                question_id = item.get("id", str(i))
                
                test_cases.append((img_path, formatted_question, correct_answer, tag, question_id))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} test cases...")
                
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue
        
        if not test_cases:
            print("No valid test cases created. Check dataset structure.")
        else:
            print(f"Successfully created {len(test_cases)} test cases for evaluation")
            
        return test_cases, list(all_tags)
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return [], [] 