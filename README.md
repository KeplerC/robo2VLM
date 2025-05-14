# Robo2VLM

This repository provides tools and resources for working with Visual Question Answering (VQA) datasets and models. It includes utilities for dataset creation, model finetuning, answer generation, and benchmarking.

## Overview

The project is organized into several key components:

*   **Dataset Creation**: Scripts and documentation for creating and managing VQA datasets, particularly for use with the Hugging Face `datasets` library.
*   **Model Finetuning**: (Placeholder for details on model finetuning capabilities)
*   **Answer Generation**: (Placeholder for details on generating answers/predictions using VQA models)
*   **Benchmarking**: (Placeholder for details on benchmarking VQA models or dataset performance)

## VQA Dataset

A core part of this project is the VQA dataset, which is collected from multimodal trajectories.

### Dataset Structure

The dataset generally includes the following fields:

*   `id`: Unique identifier for each VQA item
*   `question`: The question text
*   `choices`: List of possible answer choices
*   `correct_answer_idx`: Index of the correct answer in the `choices` list
*   `images`: List of question images (actual image data)
*   `choice_images`: List of images associated with each answer choice (actual image data)
*   `metadata`: Additional metadata about the VQA item (stored as a JSON string)

For detailed information on the dataset and how to use it, please refer to `scripts/README_dataset.md`.

## Directory Structure

*   `benchmark/`: Contains scripts and resources for benchmarking VQA models and datasets.
*   `doc/`: Intended for general project documentation.
*   `finetune/`: Contains scripts and resources for finetuning VQA models.
*   `generation/`: Contains scripts and resources for generating VQA outputs or predictions.
*   `scripts/`: Contains utility scripts, primarily for dataset creation (e.g., `create_huggingface_dataset.py` and `README_dataset.md`).

## Getting Started

1.  **Explore the Dataset**: Start by understanding the VQA dataset. Refer to `scripts/README_dataset.md` for details on its structure and usage.
2.  **Dataset Creation**: If you need to create or modify VQA datasets, explore the scripts within the `scripts/` directory.
3.  **Model Interaction**: For model finetuning, generation, or benchmarking, refer to the respective directories (`finetune/`, `generation/`, `benchmark/`). (Further documentation within these directories would be beneficial).

## Contributing

(Placeholder for contribution guidelines)

## License

(Placeholder for license information)
