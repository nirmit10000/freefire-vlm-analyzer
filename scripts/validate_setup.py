#!/usr/bin/env python3
"""
VALIDATION SCRIPT

What this does:
1. Checks if Python version is OK
2. Checks if all required packages are installed
3. Checks if GPU is available
4. Creates necessary folders
5. Tests if AI model can load

Run this BEFORE analyzing videos to catch problems early.

Usage:
    python scripts/validate_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python():
    """Check if Python version is 3.8+"""
    print("1️⃣  Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor} (need 3.8 or newer)")
        return False


def check_packages():
    """Check if required packages are installed"""
    print("\n2️⃣  Checking packages...")
    
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "tqdm": "tqdm",
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - MISSING!")
            missing.append(name)
    
    if missing:
        print(f"\n   ❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
    
    return len(missing) == 0


def check_gpu():
    """Check if GPU is available"""
    print("\n3️⃣  Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU: {gpu_name}")
            print(f"   ✓ Memory: {gpu_mem:.1f} GB")
            return True
        else:
            print("   ⚠️  No GPU detected")
            print("   Will use CPU (VERY SLOW - expect 30+ sec per frame)")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def check_folders():
    """Check and create necessary folders"""
    print("\n4️⃣  Checking folders...")
    
    from src.config import get_paths, ensure_directories
    
    # Create all folders
    ensure_directories()
    
    # Verify they exist
    paths = get_paths()
    critical = ["videos_raw", "frames", "video_analyses", "logs"]
    
    for name in critical:
        path = paths[name]
        if path.exists():
            print(f"   ✓ {name}/")
        else:
            print(f"   ✗ {name}/ - FAILED TO CREATE")
    
    return True


def check_frames():
    """Check if frames have been extracted"""
    print("\n5️⃣  Checking frames...")
    
    from src.config import get_paths
    paths = get_paths()
    frames_dir = paths["frames"]
    
    if not frames_dir.exists():
        print("   ⚠️  No frames folder yet")
        return False
    
    video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("   ⚠️  No frame folders found")
        print("   Run: python scripts/extract_frames.py --fps 1")
        return False
    
    for video_dir in video_dirs:
        frames = list(video_dir.glob("frame_*.jpg"))
        print(f"   ✓ {video_dir.name}: {len(frames)} frames")
    
    return True


def test_model():
    """Try to load the AI model"""
    print("\n6️⃣  Testing model (takes 1-2 min)...")
    
    try:
        from src.qwen_local_client import QwenLocalClient
        
        print("   Loading model...")
        client = QwenLocalClient()
        print(f"   ✓ Model loaded on {client.device}")
        
        client.cleanup()
        print("   ✓ Cleanup successful")
        return True
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("🔍 VALIDATION - CHECKING ENVIRONMENT")
    print("="*60)
    
    # Run all checks
    results = []
    results.append(("Python", check_python()))
    results.append(("Packages", check_packages()))
    results.append(("GPU", check_gpu()))
    results.append(("Folders", check_folders()))
    results.append(("Frames", check_frames()))
    results.append(("Model", test_model()))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    # Final verdict
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to analyze videos!")
        print("\nNext step:")
        print("  python scripts/analyze_video.py --video-id <VIDEO_NAME> --test")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️  SOME CHECKS FAILED")
        print("="*60)
        print("\nFix the issues above, then run validation again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())