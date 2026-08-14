#!/usr/bin/env python3
"""
Test input prompt after clicking 4 calibration points.
"""

import cv2
import sys
import os
import time

# Ensure correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("Working directory:", os.getcwd())
print()

from modules.homography import Homography

def test_input_prompt():
    """Test input prompts after calibration."""
    
    print()
    print("="*60)
    print(" TEST INPUT PROMPT")
    print("="*60)
    print()
    
    # Check image file
    image_path = "data/anh_ngatu_tinh.jpg"
    
    if not os.path.exists(image_path):
        print("[ERROR] File not found:", image_path)
        return False
    
    print("[INFO] Loading image:", image_path)
    image = cv2.imread(image_path)
    
    if image is None:
        print("[ERROR] Cannot read image")
        return False
    
    print("[OK] Image loaded")
    print()
    
    homography = Homography()
    
    print("[INFO] Instructions: Click 4 points")
    print()
    
    # Select points
    points = homography.select_points(image)
    
    if points is None:
        print("[CANCEL] User cancelled")
        return False
    
    print("[OK] Selected 4 points")
    print()
    print("[INFO] Closing window and testing input...")
    print()
    
    # Close windows and prepare for input
    cv2.destroyAllWindows()
    time.sleep(0.5)
    sys.stdout.flush()
    
    # Test input
    try:
        width = float(
            input(
                "Enter width in meters: "
            )
        )
        
        height = float(
            input(
                "Enter height in meters: "
            )
        )
        
        print()
        print("[OK] Input successful!")
        print("Width:", width, "m")
        print("Height:", height, "m")
        
        return True
        
    except ValueError:
        print("[ERROR] Invalid input")
        return False
    except KeyboardInterrupt:
        print("\n[CANCEL] User cancelled")
        return False

if __name__ == "__main__":
    
    success = test_input_prompt()
    
    print()
    print("="*60)
    
    if success:
        print("[OK] TEST PASSED")
        print("Run: python main.py")
    else:
        print("[FAIL] TEST FAILED")
    
    print("="*60)
    print()
