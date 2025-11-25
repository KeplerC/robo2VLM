# VQA Data Generation Toolkit

## Introduction

This toolkit provides a suite of tools and scripts to generate and process datasets for Visual Question Answering (VQA) applications. It allows users to leverage various data sources, including TensorFlow Datasets (TFDS) and DROID datasets, and prepare them for VQA model training and evaluation.

## Core Features

*   **Flexible Data Loading:** Ingest data from standard sources like TFDS and specialized robotics datasets like DROID.
*   **Dockerized Environment:** Ensures reproducibility and easy setup across different systems.
*   **GPU Accelerated:** Leverages NVIDIA GPUs for efficient data processing.
*   **Local LLM Integration:** Supports tools like Ollama for tasks that may require local language model inference (e.g., using Llama 3.2).
*   **Data Processing Scripts:** Includes utilities for further processing and filtering of generated VQA data.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
*   **Docker:** [Installation Guide](https://docs.docker.com/get-docker/)
*   **NVIDIA GPU:** With appropriate drivers installed.
*   **Ollama (Optional, for LLM-dependent tasks):** [Ollama Website](https://ollama.com/)

## Setup and Installation

1.  **Clone the Repository (if you haven't already):**
    ```bash
    # git clone <repository-url>
    # cd <repository-directory>/generation
    ```
    *(This README assumes you are operating from within the `generation` directory of the project.)*

2.  **Install Ollama (if needed):**
    Follow the instructions on the [Ollama website](https://ollama.com/) or run:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

3.  **Pull a Language Model (e.g., Llama 3.2, if using Ollama):**
    ```bash
    ollama pull llama3.2
    ```
    You might also want to ensure Ollama is serving if your scripts require it directly:
    ```bash
    ollama serve &
    ```

4.  **Build the Docker Image:**
    From the directory containing the `Dockerfile` (e.g., the `generation/` directory):
    ```bash
    docker build --network=host . -t dev-vqa
    ```

## Usage: Running Data Generation Tasks

The primary way to use the toolkit is by running scripts within the Docker container.

**General Docker Command Structure:**

```bash
docker run \
    --privileged \
    --net=host \
    --gpus all \
    --shm-size=10.24gb \
    -ti \
    --volume="$PWD:/app" \
    --volume="/path/to/your/raw_datasets:/raw_data" \
    --volume="/path/to/your/processed_output:/processed_data" \
    dev-vqa \
    bash -c "your_command_here"
```

**Explanation of Docker Options:**
*   `--privileged --net=host --gpus all`: Provides necessary permissions and hardware access (GPU).
*   `--shm-size=10.24gb`: Sets shared memory size, which can be important for data-intensive tasks.
*   `-ti`: Runs the container in interactive mode. Replace with `-d` for detached mode (runs in the background).
*   `--volume="$PWD:/app"`: Mounts your current working directory (expected to be the project's `generation` directory) into `/app` inside the container.
*   `--volume="/path/to/your/raw_datasets:/raw_data"`: **(Important)** Replace `/path/to/your/raw_datasets` with the actual path on your host machine where your input datasets are stored. This will be accessible at `/raw_data` inside the container.
*   `--volume="/path/to/your/processed_output:/processed_data"`: **(Important)** Replace `/path/to/your/processed_output` with the path on your host machine where you want the generated/processed data to be saved. This will be accessible at `/processed_data` inside the container.
*   `dev-vqa`: The name of the Docker image built earlier.
*   `bash -c "your_command_here"`: The command to execute inside the container.

**Example Commands:**

1.  **Loading Data from TFDS:**
    This script typically downloads and processes datasets from TensorFlow Datasets.
    ```bash
    docker run --privileged --net=host --gpus all --shm-size=10.24gb -ti \
        --volume="$PWD:/app" \
        --volume="/path/to/your/tfds_output:/processed_data" \
        dev-vqa \
        bash -c "RAY_memory_monitor_refresh_ms=0 python3 load_from_tfds.py --output_dir /processed_data"
    ```
    *Note: The `RAY_memory_monitor_refresh_ms=0` setting is often beneficial for Ray-based scripts.*
    *Adjust `/path/to/your/tfds_output` to your desired output location on your host machine.*

2.  **Loading Data from DROID:**
    This script is used for loading data from DROID-formatted datasets. It often expects input data to be available at a conventional `/data` path inside the container.
    ```bash
    docker run --privileged --net=host --gpus all --shm-size=10.24gb -ti \
        --volume="$PWD:/app" \
        --volume="/path/to/your/droid_and_rtx_datasets:/data" \
        --volume="/path/to/your/droid_processing_output:/processed_data" \
        dev-vqa \
        bash -c "RAY_memory_monitor_refresh_ms=0 python3 load_from_droid.py --output_dir /processed_data"
    ```
    *Replace `/path/to/your/droid_and_rtx_datasets` with the location of your DROID/RTX datasets on your host.*
    *Replace `/path/to/your/droid_processing_output` with your desired output location.*
    *This example assumes `load_from_droid.py` can take an `--output_dir`. If it writes to a fixed path or needs specific subdirectories in `/data`, you may need to adjust mounts or script calls.*

3.  **Processing and Filtering VQA Data:**
    The toolkit includes scripts in the `scripts/` directory for further refining your VQA datasets.
    
    To run these, you can enter an interactive session in the container:
    ```bash
    docker run --privileged --net=host --gpus all --shm-size=10.24gb -ti \
        --volume="$PWD:/app" \
        --volume="/path/to/your/data_to_process:/data_in_container" \
        dev-vqa bash
    ```
    *Replace `/path/to/your/data_to_process` with the host path containing data you want to process (e.g., a directory previously used as `/processed_data`). It will be available at `/data_in_container`.*

    Once inside the container (your command prompt will change, e.g., `root@hostname:/app#`):
    ```bash
    # Navigate to the scripts directory if needed (usually /app/scripts)
    # cd /app 

    # Example: Process data
    # python3 scripts/process_vqa_data.py --input_path /data_in_container/some_input_file --output_path /data_in_container/processed_output_file
    
    # Example: Filter data
    # python3 scripts/filter_out.py --input_path /data_in_container/some_input_file --output_path /data_in_container/filtered_output_file
    ```
    *Note: The exact arguments and behavior of `process_vqa_data.py` and `filter_out.py` are illustrative. You may need to inspect the scripts or use their help messages (e.g., `python3 scripts/your_script.py --help`) for actual usage and required input/output paths.*

## Important Considerations

*   **Dataset Paths:** Carefully replace placeholder paths like `/path/to/your/raw_datasets` with the actual paths on your host system. Incorrect volume mounts are a common source of issues.
*   **Memory for Ray:** The `RAY_memory_monitor_refresh_ms=0` environment variable is used in some commands. This can be helpful for managing memory in Ray applications.
*   **Customization:** The provided scripts and commands serve as a starting point. You may need to customize them based on your specific dataset locations, formats, and processing requirements.
*   **Script Arguments:** For detailed usage of each Python script (`load_from_tfds.py`, `load_from_droid.py`, `scripts/process_vqa_data.py`, etc.), refer to their respective command-line arguments (often accessible via a `--help` flag).
*   **Working Directory:** Ensure you run the `docker build` and `docker run` commands from the correct directory (typically the `generation` directory where the `Dockerfile` is located, and where your project files are accessible via `$PWD`).