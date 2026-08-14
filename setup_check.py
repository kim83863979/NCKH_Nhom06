#!/usr/bin/env python3
"""
🚀 QUICK START - Traffic BEV Project

Chạy file này để kiểm tra môi trường và cài đặt.
"""

import sys
import subprocess
import os
from pathlib import Path

# Đảm bảo chúng ta ở đúng thư mục
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_python():
    """Kiểm tra phiên bản Python."""
    print_header("1. KIỂM TRA PYTHON")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ yêu cầu!")
        return False
    
    print("✅ Python version OK")
    return True

def check_packages():
    """Kiểm tra các package cần thiết."""
    print_header("2. KIỂM TRA PACKAGES")
    
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        "scipy": "scipy",
        "ultralytics": "ultralytics"
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - THIẾU")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Cần cài đặt: {', '.join(missing)}")
        return False
    
    print("✅ Tất cả packages OK")
    return True

def check_data():
    """Kiểm tra dữ liệu đầu vào."""
    print_header("3. KIỂM TRA DỮ LIỆU")
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"❌ Thư mục 'data' không tồn tại")
        print("   Tạo thư mục: mkdir data")
        return False
    
    print(f"✅ Thư mục 'data' tồn tại")
    
    # Kiểm tra file video
    video_found = False
    for ext in ["mp4", "avi", "mov", "mkv"]:
        videos = list(data_dir.glob(f"*.{ext}"))
        if videos:
            print(f"✅ Video tìm thấy: {videos[0].name}")
            video_found = True
            break
    
    if not video_found:
        print("❌ KHÔNG tìm thấy video (mp4, avi, mov, mkv)")
        print("   Đặt video vào thư mục 'data/'")
        return False
    
    # Kiểm tra ảnh calibration
    if not (data_dir / "anh_ngatu_tinh.jpg").exists():
        print("⚠️  Không tìm thấy 'anh_ngatu_tinh.jpg'")
        print("   Cần thiết cho lần đầu calibrate homography")
    else:
        print("✅ Ảnh calibration tìm thấy")
    
    # Kiểm tra ảnh vệ tinh
    if not (data_dir / "anh_ve_tinh_BEV.jpg").exists():
        print("⚠️  Không tìm thấy 'anh_ve_tinh_BEV.jpg'")
        print("   Sẽ dùng nền màu xám nếu không có")
    else:
        print("✅ Ảnh vệ tinh tìm thấy")
    
    return True

def install_packages():
    """Cài đặt các packages."""
    print_header("4. CÀI ĐẶT PACKAGES")
    
    print("Chạy: pip install -r requirements.txt")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✅ Cài đặt thành công")
            return True
        else:
            print("❌ Cài đặt thất bại")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def create_weights_dir():
    """Tạo thư mục weights."""
    print_header("5. CẤU TRÚC THƯ MỤC")
    
    dirs = ["data", "weights", "modules", "utils"]
    
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"✅ {d}/")
        else:
            path.mkdir(exist_ok=True)
            print(f"✅ Tạo {d}/")
    
    return True

def main():
    """Chạy checks."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        🚀 TRAFFIC BEV PROJECT - SETUP GUIDE 🚀            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    checks = [
        ("Python", check_python()),
        ("Packages", check_packages()),
        ("Data", check_data()),
    ]
    
    all_ok = all(result for _, result in checks)
    
    print_header("📋 KẾT QUẢ KIỂM TRA")
    
    for name, result in checks:
        status = "✅ OK" if result else "❌ FAIL"
        print(f"{name:15} {status}")
    
    print("\n")
    
    if not all_ok:
        print("⚠️  CÓ VẤNĐỀ CẦN FIX:")
        print()
        
        if not check_python():
            print("1️⃣  Cập nhật Python lên 3.8+")
        
        if not check_packages():
            print("2️⃣  Chạy: pip install -r requirements.txt")
        
        if not check_data():
            print("3️⃣  Đặt video vào thư mục 'data/'")
        
        print("\n🆘 Cần cài đặt? Chạy:")
        print("   python setup_check.py --install")
        
    else:
        print("🎉 TẤT CẢ KIỂM TRA PASSED!")
        print()
        print("🚀 Ready to run:")
        print("   python main.py")
    
    print()

if __name__ == "__main__":
    
    if "--install" in sys.argv:
        install_packages()
    else:
        main()
