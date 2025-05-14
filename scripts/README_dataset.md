# VQA Dataset

This dataset contains Visual Question Answering (VQA) examples collected from multimodal trajectories. The dataset includes questions, multiple-choice answers, and associated images. All images are stored directly in the dataset.

## Dataset Structure

The dataset contains the following fields:

- `id`: Unique identifier for each VQA item
- `question`: The question text
- `choices`: List of possible answer choices
- `correct_answer_idx`: Index of the correct answer in the choices list
- `images`: List of question images as actual image data (not just paths)
- `choice_images`: List of images associated with each answer choice as actual image data
- `metadata`: Additional metadata about the VQA item (stored as JSON string)

## Usage

You can load the dataset using the HuggingFace datasets library:

```python
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt

# Load from HuggingFace Hub
dataset = load_dataset("your-username/dataset-name")
dataset = load_from_disk("hf_dataset")
# Example: Get a sample item
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Choices: {sample['choices']}")
print(f"Correct answer: {sample['choices'][sample['correct_answer_idx']]}")

# Display an image associated with the question
if sample['images']:
    plt.figure(figsize=(10, 8))
    plt.imshow(sample['images'][0])
    plt.title(f"Question: {sample['question']}")
    plt.axis('off')
    plt.show()
```

## Dataset Creation

This dataset was created by:
1. Extracting VQA items from trajectory data
2. Randomly sampling a subset of the items
3. Loading and embedding actual image data directly in the dataset
4. Converting to the HuggingFace datasets format

