#!/usr/bin/env python3
"""
FRAME EXTRACTION SCRIPT

What this does:
1. Looks in data/videos_raw/ for .mp4 files
2. Extracts 1 frame per second from each video
3. Saves frames as .jpg in data/frames/<video_name>/

Usage:
    python scripts/extract_frames.py --fps 1
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so we can import src.*
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our code
from src.config import get_paths
from src.frame_extractor import extract_frames_from_video


def main():
    """Main function that runs when you execute this script"""
    
    # Get folder paths
    paths = get_paths()
    videos_dir = paths["videos_raw"]
    frames_root = paths["frames"]
    
    # Make sure frames folder exists
    frames_root.mkdir(parents=True, exist_ok=True)
    
    # Find all video files in videos_raw/
    video_files = [
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in ['.mp4', '.mov', '.mkv', '.avi']
    ]
    
    # Check if we found any videos
    if not video_files:
        print("="*60)
        print("⚠️  NO VIDEOS FOUND!")
        print("="*60)
        print(f"Looking in: {videos_dir}")
        print("\nPlease put your .mp4 videos in that folder, then run again.")
        return
    
    print("="*60)
    print("🎬 FRAME EXTRACTION STARTING")
    print("="*60)
    print(f"Found {len(video_files)} video(s)")
    print()
    
    # Process each video
    for video_file in video_files:
        # Get video name without extension
        video_name = video_file.stem  # e.g., "5ap2epx9kdjx1g"
        
        # Create folder for this video's frames
        video_frames_dir = frames_root / video_name
        
        print(f"Processing: {video_file.name}")
        print(f"Output to: {video_frames_dir}")
        print()
        
        try:
            # Extract frames (1 per second by default)
            extract_frames_from_video(
                video_path=video_file,
                output_dir=video_frames_dir,
                fps=1  # 1 frame per second
            )
            print()
        except Exception as e:
            print(f"❌ ERROR: {e}")
            print()
    
    print("="*60)
    print("✅ FRAME EXTRACTION COMPLETE")
    print("="*60)
    print(f"Check frames in: {frames_root}")


if __name__ == "__main__":
    main()