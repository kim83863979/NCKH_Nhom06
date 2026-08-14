#!/usr/bin/env bash

# 🚀 TRAFFIC BEV PROJECT - QUICK START SCRIPT

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Traffic BEV Project - Installation & Quick Start"
echo "════════════════════════════════════════════════════════"
echo ""

# 1. Kiểm tra Python
echo "✓ Checking Python version..."
python3 --version
echo ""

# 2. Tạo virtual environment (tuỳ chọn)
if [ "$1" == "--venv" ]; then
    echo "✓ Creating virtual environment..."
    python3 -m venv venv
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "✓ Virtual environment activated"
    echo ""
fi

# 3. Upgrade pip
echo "✓ Upgrading pip..."
python3 -m pip install --upgrade pip
echo ""

# 4. Cài đặt requirements
echo "✓ Installing requirements..."
pip install -r requirements.txt
echo ""

# 5. Kiểm tra cài đặt
echo "✓ Running setup check..."
python3 setup_check.py
echo ""

echo "════════════════════════════════════════════════════════"
echo "🎉 Setup Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📖 Tiếp theo:"
echo "   1. Đặt video vào thư mục 'data/'"
echo "   2. Chạy: python main.py"
echo ""
echo "📚 Tài liệu: Xem SETUP.md"
echo ""
