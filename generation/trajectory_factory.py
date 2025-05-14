import os
import random
import subprocess
from typing import Dict, Optional, Union
from pathlib import Path

from trajectory import BaseTrajectory
from tfds_trajectory import TFDSTrajectory 
from droid_trajectory import DROIDTrajectory

def create_trajectory(
    episode_data: Dict, 
    output_dir: Optional[Union[str, Path]] = None,
    download_raw: bool = True,
    temp_base_dir: str = "./droid_raw"
) -> BaseTrajectory:
    """
    Factory function to create the appropriate trajectory implementation
    based on the provided data and flags.
    
    Args:
        episode_data: Dictionary containing TFDS episode metadata
        output_dir: Optional output directory for processed data
        download_raw: Whether to download and process raw data (if False, use TFDS-only)
        temp_base_dir: Base directory for downloaded raw data
        
    Returns:
        A BaseTrajectory implementation (either TFDSTrajectory or DROIDTrajectory)
    """
    # Use TFDS-only trajectory if raw data download is disabled
    if not download_raw:
        return TFDSTrajectory(episode_data, output_dir)
    
    # Otherwise, try to create a full DROIDTrajectory with raw data
    try:
        # Extract file path
        file_path = episode_data["episode_metadata"]["file_path"]
        
        # Extract episode ID
        import re
        episode_id_match = re.search(r'([^/]+)/trajectory\.h5$', file_path)
        if episode_id_match:
            episode_id = episode_id_match.group(1)
        else:
            episode_id = f"episode_{random.randint(1000, 9999)}"
        
        # Determine data paths
        temp_dir = os.path.join(temp_base_dir, episode_id)
        
        # Extract GS path for download
        path_parts = file_path.replace("/trajectory.h5", "").split('/')
        try:
            base_index = path_parts.index("droid_raw")
            if path_parts[base_index+1] != '1.0.1':
                raise ValueError("Found 'droid_raw' but not '1.0.1' following it.")
            episode_folder = "/".join(path_parts[base_index+2:])
        except (ValueError, IndexError):
            if len(path_parts) < 4:
                raise ValueError(f"Path too short for fallback: {file_path}")
            episode_folder = "/".join(path_parts[-4:])

        gs_parent_dir = f"gs://gresearch/robotics/droid_raw/1.0.1/{episode_folder}/"
        
        # Download raw data
        os.makedirs(temp_dir, exist_ok=True)
        try:
            result = subprocess.run(["gsutil", "-m", "cp", "-r", gs_parent_dir, temp_dir], 
                                    capture_output=True)
        except Exception as e:
            print(f"Error downloading data: {e}")
            # Fall back to TFDS-only trajectory on download failure
            return TFDSTrajectory(episode_data, output_dir)
        
        # Construct expected local path to trajectory file
        downloaded_content_dir_name = gs_parent_dir.strip('/').split('/')[-1]
        local_scene_path = os.path.join(temp_dir, downloaded_content_dir_name)
        
        # Create DROIDTrajectory instance
        return DROIDTrajectory(
            episode_data=episode_data,
            scene_path=local_scene_path,
            output_dir=output_dir,
            gsutil_path=gs_parent_dir
        )
        
    except Exception as e:
        print(f"Error creating DROIDTrajectory: {e}")
        # Fall back to TFDS-only trajectory on any error
        return TFDSTrajectory(episode_data, output_dir) 