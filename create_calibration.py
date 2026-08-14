#!/usr/bin/env python3
"""
Tạo ảnh calibration từ video.

Script này sẽ:
1. Đọc frame đầu tiên từ myDemoVideo.mp4
2. Lưu làm anh_ngatu_tinh.jpg
"""

import cv2
import os
import sys

# Đảm bảo chúng ta ở đúng thư mục
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

def create_calibration_image():
    """Tạo ảnh calibration từ video."""
    
    video_path = "data/myDemoVideo.mp4"
    output_path = "data/anh_ngatu_tinh.jpg"
    
    # Kiểm tra video có tồn tại không
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return False
    
    # Kiểm tra ảnh đã tồn tại không
    if os.path.exists(output_path):
        print(f"✅ Ảnh calibration đã tồn tại: {output_path}")
        return True
    
    print(f"📹 Đang đọc video: {video_path}")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Không thể mở video")
        return False
    
    # Đọc frame đầu tiên
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("❌ Không thể đọc frame từ video")
        cap.release()
        return False
    
    # Resize cho nhỏ hơn một chút (tuỳ chọn)
    height, width = frame.shape[:2]
    print(f"   Kích thước video: {width}x{height}")
    
    # Lưu frame
    cv2.imwrite(output_path, frame)
    
    cap.release()
    
    print(f"✅ Ảnh calibration đã được tạo: {output_path}")
    print()
    print("📋 Hướng dẫn sử dụng:")
    print("   1. Chạy: python main.py")
    print("   2. Cửa sổ 'CHON 4 DIEM HOMOGRAPHY' sẽ xuất hiện")
    print("   3. Click 4 điểm tạo thành tứ giác trên mặt đường:")
    print("      - Điểm 1: Trên - Trái")
    print("      - Điểm 2: Trên - Phải")
    print("      - Điểm 3: Dưới - Phải")
    print("      - Điểm 4: Dưới - Trái")
    print("   4. Nhấn ENTER để xác nhận")
    print("   5. Nhập kích thước thực tế (mét)")
    print()
    
    return True

if __name__ == "__main__":
    
    print()
    print("="*60)
    print(" TẠO ẢNH CALIBRATION TỪ VIDEO")
    print("="*60)
    print()
    
    success = create_calibration_image()
    
    if success:
        print("🎉 Sẵn sàng chạy main.py!")
    else:
        print("❌ Có lỗi, vui lòng kiểm tra file video")
    
    print()
