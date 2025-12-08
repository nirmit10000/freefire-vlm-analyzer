"""
Configuration for the analyzer.
This file manages all paths and settings.
"""

import os
from pathlib import Path

# Find the project root folder (freefire-vlm-analyzer/)
# This works no matter where Python runs from
BASE_DIR = Path(__file__).resolve().parent.parent


def get_paths():
    """
    Returns all important folder paths.
    
    Returns a dictionary like:
    {
        "base_dir": /home/almalinux/freefire-vlm-analyzer,
        "videos_raw": /home/almalinux/freefire-vlm-analyzer/data/videos_raw,
        ...
    }
    """
    data_dir = BASE_DIR / "data"
    
    return {
        "base_dir": BASE_DIR,
        "data_dir": data_dir,
        "videos_raw": data_dir / "videos_raw",
        "frames": data_dir / "frames",
        "outputs": data_dir / "outputs",
        "video_analyses": data_dir / "outputs" / "video_analyses",
        "logs": data_dir / "outputs" / "logs",
    }


def get_model_config():
    """
    Returns model settings.
    
    Returns:
    {
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "max_tokens": 512,
        "temperature": 0.1
    }
    """
    return {
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "max_tokens": 512,
        "temperature": 0.1,  # Low = more predictable JSON output
    }


def get_processing_config():
    """
    Returns processing settings.
    
    Returns:
    {
        "frame_sample_interval": 5,  # Process every 5th frame
        "max_frames_per_video": 200, # Maximum 200 frames per video
        "retry_count": 2,             # Retry failed frames 2 times
        "skip_on_error": True         # Skip frame if all retries fail
    }
    """
    return {
        "frame_sample_interval": 5,   # Process every 5th frame (5 second intervals)
        "max_frames_per_video": 200,  # Cap at 200 frames (prevents huge videos taking forever)
        "retry_count": 2,              # Retry 2 times if frame fails
        "skip_on_error": True,         # Skip and continue if frame permanently fails
    }


def ensure_directories():
    """
    Creates all necessary folders if they don't exist.
    Safe to call multiple times - won't break if folders exist.
    """
    paths = get_paths()
    
    # List of all folders we need
    folders_to_create = [
        paths["data_dir"],
        paths["videos_raw"],
        paths["frames"],
        paths["outputs"],
        paths["video_analyses"],
        paths["logs"],
    ]
    
    # Create each folder
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)
    
    print("✓ All folders created/verified")