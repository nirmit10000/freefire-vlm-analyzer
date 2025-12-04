# src/analyzer.py

import os
import json
from typing import Optional, List

from .config import get_paths
from .prompts import BASE_SYSTEM_PROMPT, get_frame_analysis_prompt
from .qwen_client import analyze_frame


def list_frame_files(frames_dir: str) -> List[str]:
    """
    Returns a sorted list of frame image file paths inside frames_dir.
    Expected format: frame_00000.jpg, frame_00001.jpg, etc.
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    files = [
        f
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files.sort()
    return [os.path.join(frames_dir, f) for f in files]


def analyse_single_frame_to_json(
    image_path: str,
    output_path: str,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> None:
    """
    Sends one frame to the VLM and saves the model's response as JSON.

    If the model output is not valid JSON, it is stored under the key
    "raw_text" along with an "error" field.
    """
    if system_prompt is None:
        system_prompt = BASE_SYSTEM_PROMPT

    if user_prompt is None:
        user_prompt = get_frame_analysis_prompt()

    # Call the model
    raw_response = analyze_frame(
        image_path=image_path,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Try to parse as JSON
    data = None
    error_msg = None

    try:
        data = json.loads(raw_response)
    except Exception as e:
        error_msg = str(e)
        data = {
            "raw_text": raw_response,
            "error": "Failed to parse model output as JSON",
            "exception": error_msg,
        }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save pretty-printed JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def analyse_video_frames(
    video_name: str,
    max_frames: Optional[int] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> None:
    """
    Analyse all frames for a given video (by its base name).

    Example:
      video_name = "match1"
      Frames directory: data/frames/match1
      Output directory: data/outputs/raw_model/match1

    Parameters
    ----------
    video_name : str
        Base name of the video (without extension).
    max_frames : Optional[int]
        If given, only the first `max_frames` frames are analysed.
    temperature : float
        Model sampling temperature.
    max_tokens : int
        Maximum tokens to generate per frame.
    """
    paths = get_paths()

    frames_dir = os.path.join(paths["frames"], video_name)
    output_dir = os.path.join(paths["outputs_raw"], video_name)

    frame_files = list_frame_files(frames_dir)
    if max_frames is not None:
        frame_files = frame_files[:max_frames]

    if not frame_files:
        print(f"[INFO] No frame images found for video '{video_name}' in: {frames_dir}")
        return

    print(f"[INFO] Analysing {len(frame_files)} frame(s) for video: {video_name}")
    print(f"[INFO] Saving JSON outputs to: {output_dir}")

    for image_path in frame_files:
        frame_filename = os.path.basename(image_path)
        base_name, _ = os.path.splitext(frame_filename)
        out_path = os.path.join(output_dir, f"{base_name}.json")

        print(f"[INFO] Analysing frame: {frame_filename}")
        try:
            analyse_single_frame_to_json(
                image_path=image_path,
                output_path=out_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"[ERROR] Failed to analyse frame {frame_filename}: {e}")
