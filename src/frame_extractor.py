# src/frame_extractor.py

import os
from typing import List
import cv2
from tqdm import tqdm


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: int = 1,
) -> List[str]:
    """
    Extracts frames from a video file at a target sampling rate
    (e.g., 1 frame per second) and saves them as .jpg images.

    Parameters
    ----------
    video_path : str
        Path to the input video file (e.g., .mp4).
    output_dir : str
        Folder where extracted frames will be stored.
    fps : int
        How many frames per second to sample (e.g., 1 means 1 frame per second).

    Returns
    -------
    List[str]
        List of file paths to the saved frame images.
    """
    if fps <= 0:
        fps = 1  # safety fallback

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If original FPS is unknown or zero, fall back to sampling every frame_interval frames
    if original_fps and original_fps > 0:
        # example: original_fps=30, fps=1  => frame_interval ~ 30
        frame_interval = max(int(round(original_fps / fps)), 1)
    else:
        frame_interval = 30  # arbitrary fallback

    frame_paths: List[str] = []
    frame_idx = 0
    saved_idx = 0

    for _ in tqdm(range(total_frames), desc=f"Extracting {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Save every N-th frame
        if frame_idx % frame_interval == 0:
            filename = f"frame_{saved_idx:05d}.jpg"
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, frame)
            frame_paths.append(out_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return frame_paths
