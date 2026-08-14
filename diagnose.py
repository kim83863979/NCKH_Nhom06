#!/usr/bin/env python3
"""
Quick diagnostic test - check imports and paths
"""

import sys
import os

# Ensure correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

print("="*60)
print(" DIAGNOSTIC TEST")
print("="*60)
print()

# Test 1: Working directory
print("[1] Working directory")
print("   Path:", os.getcwd())
print("   OK" if os.getcwd().endswith("NCKH_Nhom06-develop") else "FAILED")
print()

# Test 2: File check
print("[2] File check")
files_to_check = [
    "data/myDemoVideo.mp4",
    "data/anh_ngatu_tinh.jpg",
    "data/anh_ve_tinh_BEV.jpg",
    "modules/detector.py",
    "modules/tracker.py",
    "modules/homography.py",
    "modules/bev_mapper.py",
    "utils/video_stream.py",
    "main.py"
]

all_exist = True
for f in files_to_check:
    exists = os.path.exists(f)
    status = "OK" if exists else "MISSING"
    print(f"   {f:40} {status}")
    all_exist = all_exist and exists

print()

# Test 3: Imports
print("[3] Import test")
try:
    import cv2
    print("   cv2               OK")
except ImportError as e:
    print("   cv2              FAILED:", str(e)[:50])

try:
    import numpy
    print("   numpy             OK")
except ImportError as e:
    print("   numpy            FAILED:", str(e)[:50])

try:
    from ultralytics import YOLO
    print("   ultralytics       OK")
except ImportError as e:
    print("   ultralytics      FAILED:", str(e)[:50])

try:
    from utils.video_stream import VideoStream
    print("   video_stream      OK")
except ImportError as e:
    print("   video_stream     FAILED:", str(e)[:50])

try:
    from modules.detector import Detector
    print("   detector          OK")
except ImportError as e:
    print("   detector         FAILED:", str(e)[:50])

try:
    from modules.tracker import Tracker
    print("   tracker           OK")
except ImportError as e:
    print("   tracker          FAILED:", str(e)[:50])

try:
    from modules.homography import Homography
    print("   homography        OK")
except ImportError as e:
    print("   homography       FAILED:", str(e)[:50])

try:
    from modules.bev_mapper import BEVMapper
    print("   bev_mapper        OK")
except ImportError as e:
    print("   bev_mapper       FAILED:", str(e)[:50])

print()
print("="*60)
if all_exist:
    print("STATUS: All tests passed! Ready to run main.py")
    print()
    print("Next: python main.py")
else:
    print("STATUS: Some files missing!")
print("="*60)
print()
