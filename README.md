# NCKH_Nhom06

Traffic_BEV_Project/
│
├── data/ # Thư mục chứa nguyên liệu đầu vào (Không chứa code)
│ ├── video_giao_thong.mp4 # Khôi dùng để đọc luồng
│ ├── anh_ngatu_tinh.jpg # Kiệt dùng để click 4 điểm
│ └── anh_ve_tinh_BEV.jpg # Khôi & Kiệt dùng làm nền bản đồ Radar
│
├── weights/ # Thư mục chứa trọng số mô hình AI
│ └── yolov8_obb.pt # Hạo lưu file model đã train tại đây
│
├── modules/ # NƠI CHỨA CODE CỐT LÕI CỦA TỪNG THÀNH VIÊN
│ ├── **init**.py  
│ ├── detector.py # File code của HẠO
│ ├── tracker.py # File code của KIM
│ ├── homography.py # File code của KIỆT (Phần 1)
│ └── bev_mapper.py # File code của KIỆT (Phần 2)
│
├── utils/ # Thư mục chứa các hàm hỗ trợ phụ
│ ├── **init**.py
│ └── video_stream.py # Khôi code phần xử lý đa luồng (threading)
│
└── main.py # File chạy chính của KHÔI (Nơi lắp ráp toàn bộ)
