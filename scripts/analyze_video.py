#!/usr/bin/env python3
"""
🎮 MAIN ANALYSIS SCRIPT

What this does:
1. Loads frames for a video
2. Samples every 5th frame (to speed up)
3. Sends each frame to AI model
4. Collects all responses
5. Saves everything to one big JSON file

Usage:
    # Test mode (only 10 frames)
    python scripts/analyze_video.py --video-id 8zpr6zukq5a8xr --test
    
    # Full analysis
    python scripts/analyze_video.py --video-id 8zpr6zukq5a8xr
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import our code
from src.config import get_paths, get_model_config, get_processing_config
from src.qwen_local_client import QwenLocalClient
from src.prompts import get_analysis_prompt
from src.json_cleaner import clean_model_output, create_error_response


def get_frames_to_process(video_id, sample_every=5, max_frames=200, test_mode=False):
    """
    Get list of frames to analyze.
    
    Args:
        video_id: Video folder name (e.g., "8zpr6zukq5a8xr")
        sample_every: Process every Nth frame (5 = every 5th frame)
        max_frames: Maximum frames to process (200 = cap at 200)
        test_mode: If True, only return 10 frames
    
    Returns:
        list: Frame file paths to process
    
    Example:
        Video has 619 frames
        sample_every=5 → selects frames 0, 5, 10, 15, ... = 124 frames
        max_frames=200 → all 124 frames (under cap)
        test_mode=True → only first 10 frames
    """
    paths = get_paths()
    frames_dir = paths["frames"] / video_id
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")
    
    # Get all frame files, sorted
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    
    if not all_frames:
        raise FileNotFoundError(f"No frames in: {frames_dir}")
    
    # Sample every Nth frame
    sampled = all_frames[::sample_every]
    
    # Apply limits
    if test_mode:
        selected = sampled[:10]
    elif max_frames:
        selected = sampled[:max_frames]
    else:
        selected = sampled
    
    return selected


def analyze_one_frame(client, frame_path, prompt, max_tokens=512, temperature=0.1, retry_count=2):
    """
    Analyze a single frame with retry logic.
    
    Args:
        client: QwenLocalClient instance
        frame_path: Path to frame image
        prompt: What to ask AI
        max_tokens: Response length limit
        temperature: Randomness (0.1 = predictable)
        retry_count: How many times to retry on failure
    
    Returns:
        dict: Analysis result (or error response)
    """
    # Try multiple times
    for attempt in range(retry_count + 1):
        try:
            # Ask AI to analyze frame
            raw_output = client.run_inference(
                image_path=str(frame_path),
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Clean and parse JSON
            result = clean_model_output(raw_output)
            
            # Add metadata
            result["frame_filename"] = frame_path.name
            result["processing_timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            # If not last attempt, try again
            if attempt < retry_count:
                continue
            else:
                # All attempts failed, return error
                return create_error_response(str(e))


def aggregate_results(frame_results):
    """
    Combine all frame results into summary statistics.
    
    Args:
        frame_results: List of individual frame analyses
    
    Returns:
        dict: Aggregated stats
    """
    # Separate successful vs failed
    successful = [r for r in frame_results if not r.get("error", False)]
    failed = [r for r in frame_results if r.get("error", False)]
    
    # Count things across all frames
    total_kills = sum(
        r.get("attributes", {}).get("eliminations_count", 0)
        for r in successful
    )
    
    total_deaths = sum(
        r.get("attributes", {}).get("player_deaths_count", 0)
        for r in successful
    )
    
    return {
        "total_frames_processed": len(frame_results),
        "successful_analyses": len(successful),
        "failed_analyses": len(failed),
        "total_eliminations": total_kills,
        "total_deaths": total_deaths,
    }


def main():
    """Main function - runs when you execute this script"""
    
    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Free Fire video frames")
    parser.add_argument("--video-id", required=True, help="Video folder name")
    parser.add_argument("--test", action="store_true", help="Test mode (10 frames only)")
    args = parser.parse_args()
    
    video_id = args.video_id
    
    # Print header
    print("="*60)
    print("🎮 FREE FIRE ANALYZER - STARTING")
    print("="*60)
    print(f"Video ID: {video_id}")
    print(f"Test Mode: {args.test}")
    print()
    
    # Get settings
    paths = get_paths()
    model_config = get_model_config()
    proc_config = get_processing_config()
    
    # Get frames to process
    try:
        frames = get_frames_to_process(
            video_id=video_id,
            sample_every=proc_config["frame_sample_interval"],
            max_frames=proc_config["max_frames_per_video"],
            test_mode=args.test
        )
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 1
    
    print(f"📊 Found {len(frames)} frames to process")
    print()
    
    # Load AI model
    print("🤖 Loading AI model...")
    try:
        client = QwenLocalClient()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    print()
    
    # Get analysis prompt
    prompt = get_analysis_prompt()
    
    # Analyze each frame
    print("🎬 Analyzing frames...")
    results = []
    
    for frame_path in tqdm(frames, desc="Processing"):
        result = analyze_one_frame(
            client=client,
            frame_path=frame_path,
            prompt=prompt,
            max_tokens=model_config["max_tokens"],
            temperature=model_config["temperature"],
            retry_count=proc_config["retry_count"]
        )
        results.append(result)
    
    print()
    
    # Aggregate stats
    print("📈 Calculating statistics...")
    stats = aggregate_results(results)
    
    # Build final output
    output = {
        "video_id": video_id,
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_config["model_name"],
            "test_mode": args.test,
        },
        "aggregated_statistics": stats,
        "frame_analyses": results,
    }
    
    # Save to file
    output_dir = paths["video_analyses"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{video_id}_analysis.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print()
    print("="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"Success: {stats['successful_analyses']}/{stats['total_frames_processed']} frames")
    print(f"Kills: {stats['total_eliminations']}")
    print(f"Deaths: {stats['total_deaths']}")
    
    # Cleanup
    client.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())