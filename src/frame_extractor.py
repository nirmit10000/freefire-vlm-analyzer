"""
This file extracts frames (screenshots) from video files.
Takes a .mp4 video, saves individual .jpg images.
"""

import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, fps=1):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file (e.g., "data/videos_raw/video.mp4")
        output_dir: Where to save frames (e.g., "data/frames/video/")
        fps: How many frames per second to extract
             fps=1 means 1 frame every second
             fps=0.5 means 1 frame every 2 seconds
    
    Returns:
        list: Paths to all extracted frame images
    
    Example:
        Video is 100 seconds long, fps=1
        Result: 100 frames saved as frame_00000.jpg to frame_00099.jpg
    """
    # Make sure fps is at least 1
    if fps <= 0:
        fps = 1
    
    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # Video's original frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in video
    
    print(f"Video: {video_path.name}")
    print(f"  Original FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    # Calculate: extract every Nth frame
    # Example: video is 30 fps, we want 1 fps → extract every 30th frame
    if original_fps and original_fps > 0:
        frame_interval = max(int(round(original_fps / fps)), 1)
    else:
        frame_interval = 30  # Default if video fps unknown
    
    print(f"  Extracting every {frame_interval} frames (target: {fps} FPS)")
    
    # Storage for saved frame paths
    saved_frames = []
    current_frame_num = 0  # Current frame we're reading
    saved_frame_num = 0    # How many frames we've saved
    
    # Progress bar (shows pretty progress in terminal)
    pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.name}", unit="frames")
    
    # Read video frame by frame
    while True:
        # Read next frame
        success, frame = cap.read()
        
        # If no more frames, stop
        if not success:
            break
        
        # Should we save this frame?
        if current_frame_num % frame_interval == 0:
            # Create filename: frame_00000.jpg, frame_00001.jpg, etc.
            filename = f"frame_{saved_frame_num:05d}.jpg"
            output_path = output_dir / filename
            
            # Save frame as image
            cv2.imwrite(str(output_path), frame)
            saved_frames.append(output_path)
            saved_frame_num += 1
        
        current_frame_num += 1
        pbar.update(1)
    
    # Clean up
    pbar.close()
    cap.release()
    
    print(f"✓ Extracted {len(saved_frames)} frames")
    return saved_frames