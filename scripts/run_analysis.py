# scripts/run_analysis.py

import os
import sys
import argparse

# Make sure Python can find the "src" package when running this script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_paths
from src.analyzer import analyse_video_frames


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL-2B-Instruct analysis on frames of one or more videos. "
            "Frames are expected in data/frames/<video_name>/ "
            "and outputs will be written to data/outputs/raw_model/<video_name>/."
        )
    )

    parser.add_argument(
        "--video-name",
        type=str,
        help=(
            "Base name of a single video to analyse (without extension). "
            "Example: if your frames are in data/frames/match1, use --video-name match1."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse all videos that have frame folders in data/frames.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally limit the number of frames per video (e.g., 50).",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model (default: 0.2).",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per frame (default: 512).",
    )

    args = parser.parse_args()
    paths = get_paths()

    frames_root = paths["frames"]

    # Decide which videos to process
    video_names = []

    if args.video_name:
        video_names.append(args.video_name)

    if args.all:
        if not os.path.isdir(frames_root):
            print(f"[ERROR] Frames root directory does not exist: {frames_root}")
            return

        for name in os.listdir(frames_root):
            candidate_dir = os.path.join(frames_root, name)
            if os.path.isdir(candidate_dir):
                video_names.append(name)

    # Remove duplicates
    video_names = sorted(set(video_names))

    if not video_names:
        print("[ERROR] No videos specified for analysis.")
        print("Use either:")
        print("  --video-name <name>   (for a single video)")
        print("or:")
        print("  --all                 (for all videos with frames)")
        return

    # Run analysis for each video
    for video_name in video_names:
        print(f"[INFO] === Analysing video: {video_name} ===")
        try:
            analyse_video_frames(
                video_name=video_name,
                max_frames=args.max_frames,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"[ERROR] Failed to analyse video '{video_name}': {e}")


if __name__ == "__main__":
    main()
