# scripts/extract_frames.py

import os
import sys
import argparse

# Make sure Python can find the "src" package when running this script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_paths
from src.frame_extractor import extract_frames_from_video


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from all videos in data/videos_raw."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Number of frames to sample per second from each video (default: 1).",
    )
    args = parser.parse_args()

    paths = get_paths()
    videos_dir = paths["videos_raw"]
    frames_root = paths["frames"]

    # Ensure the frames root directory exists
    os.makedirs(frames_root, exist_ok=True)

    # List all files in videos_raw
    video_files = [
        f
        for f in os.listdir(videos_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv"))
    ]

    if not video_files:
        print(f"[INFO] No video files found in: {videos_dir}")
        print("Place your Free Fire videos there, then run this script again.")
        return

    for fname in video_files:
        video_path = os.path.join(videos_dir, fname)
        video_name, _ = os.path.splitext(fname)

        # Each video gets its own subfolder inside data/frames
        video_frame_dir = os.path.join(frames_root, video_name)
        print(f"[INFO] Extracting frames for {fname} → {video_frame_dir}")

        try:
            extract_frames_from_video(
                video_path=video_path,
                output_dir=video_frame_dir,
                fps=args.fps,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")


if __name__ == "__main__":
    main()
