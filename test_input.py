#!/usr/bin/env python3
"""
Test input prompt sau khi click 4 diem.

Script nay se:
1. Yeu cau click 4 diem (dung anh calibration)
2. Kiem tra xem prompt "Nhap chieu rong" co hoat dong khong
"""

import cv2
import sys
import os
import time

# Damsac chung ta o dung thu muc
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print(f"Working directory: {os.getcwd()}")
print()

# Import module
from modules.homography import Homography

def test_input_prompt():
    """Test yeu cau input sau calibration."""
    
    print()
    print("="*60)
    print(" TEST INPUT PROMPT")
    print("="*60)
    print()
    
    # Kiem tra file
    image_path = "data/anh_ngatu_tinh.jpg"
    
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return False
    
    print(f"[INFO] Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("[ERROR] Failed to read image")
        return False
    
    print("[OK] Image loaded")
    print()
    
    # Tao homography
    homography = Homography()
    
    print("[INFO] Instructions: Click 4 points on the road")
    print("   (TL, TR, BR, BL)")
    print()
    
    # Select points
    points = homography.select_points(image)
    
    if points is None:
        print("[CANCEL] User cancelled")
        return False
    
    print(f"[OK] Selected 4 points")
    print()
    
    # QUAN TRONG: Dong window va chuan bi nhap
    print("[INFO] Closing window...")
    
    cv2.destroyAllWindows()
    time.sleep(0.5)
    sys.stdout.flush()
    
    print()
    
    # TEST INPUT
    try:
        # Nhap chieu rong
        width_str = input("Nhap chieu rong that te (m): ")
        width = float(width_str)
        
        print()
        
        # Nhap chieu dai
        height_str = input("Nhap chieu dai that te (m): ")
        height = float(height_str)
        
        print()
        print("[OK] Input successful!")
        print(f"    Chieu rong: {width} m")
        print(f"    Chieu dai: {height} m")
        
        return True
        
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
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
        print()
        print("Now run: python main.py")
    else:
        print("[FAIL] TEST FAILED")
    
    print("="*60)
    print()
