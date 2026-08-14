# Traffic BEV Project

Hệ thống theo dõi phương tiện giao thông với chuyển đổi từ góc nhìn camera sang bản đồ BEV (Bird's Eye View).

## 📋 Mô tả dự án

**Mục tiêu:** Xây dựng hệ thống đa chức năng chứng minh:

1. **Màn hình LEFT (Mắt thần AI):** Video gốc với các khung xoay OBB + ID phương tiện
2. **Màn hình RIGHT (Bản sao số BEV):** Bản đồ vệ tinh với các biểu tượng xe di chuyển mượt mà

## 📁 Cấu trúc thư mục

```
Traffic_BEV_Project/
│
├── data/                       # Dữ liệu đầu vào (video, ảnh)
│   ├── myDemoVideo.mp4        # Video giao thông
│   ├── anh_ngatu_tinh.jpg     # Ảnh tĩnh để calibrate homography
│   └── anh_ve_tinh_BEV.jpg    # Ảnh vệ tinh làm nền BEV
│
├── weights/                    # Trọng số mô hình AI
│   └── yolov8_obb.pt          # YOLOv8-OBB model (auto download)
│
├── modules/                    # Code cốt lõi
│   ├── __init__.py
│   ├── detector.py             # YOLOv8-OBB detection (HẠO)
│   ├── tracker.py              # BoT-SORT tracking (KIM)
│   ├── homography.py           # Calibration homography (KIỆT - Phần 1)
│   └── bev_mapper.py           # BEV transformation (KIỆT - Phần 2)
│
├── utils/                      # Hàm hỗ trợ
│   ├── __init__.py
│   └── video_stream.py         # Multi-threading video reader (KHÔI)
│
├── main.py                     # File chạy chính (KHÔI - orchestration)
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## ⚙️ Cài đặt

### 1. Cài đặt Python 3.8+

```bash
python --version  # Kiểm tra phiên bản
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Hoặc cài thủ công:**

```bash
pip install opencv-python numpy ultralytics scipy torch torchvision
```

### 3. Chuẩn bị dữ liệu

Đặt các file sau vào thư mục `data/`:

- `myDemoVideo.mp4` - Video giao thông (bắt buộc)
- `anh_ngatu_tinh.jpg` - Ảnh tĩnh để chọn 4 điểm calibrate (bắt buộc lần đầu)
- `anh_ve_tinh_BEV.jpg` - Ảnh vệ tinh nền BEV (tuỳ chọn)

## 🚀 Chạy chương trình

### Chạy chính thức

```bash
python main.py
```

### Lần đầu tiên

1. **Chương trình sẽ yêu cầu calibrate homography:**
   - Cửa sổ "CHON 4 DIEM HOMOGRAPHY" xuất hiện
   - Click lần lượt 4 điểm tạo hình tứ giác trên mặt đường:
     - **Điểm 1:** Trên - Trái
     - **Điểm 2:** Trên - Phải
     - **Điểm 3:** Dưới - Phải
     - **Điểm 4:** Dưới - Trái
   - Nhấn **ENTER** để xác nhận

2. **Nhập kích thước thực tế:**
   - Chiều rộng (m): Khoảng cách thực tế giữa điểm 1-2
   - Chiều dài (m): Khoảng cách thực tế giữa điểm 1-4

3. **Ma trận homography được lưu:** `modules/homography.npy`

### Lần tiếp theo

- Chương trình sẽ tự động dùng ma trận đã lưu
- Nếu muốn recalibrate, xóa file `modules/homography.npy`

## 🎥 Giao diện khi chạy

```
┌─────────────────────┬──────────────────────┐
│  MAT THAN AI        │  BAN SAO SO VE TINH  │
│  (Video + OBB+ID)   │  (BEV + Vehicles)    │
│                     │                      │
│  ◊─────────────────►│  ●  (Vehicles)       │
│  ┌─────┐            │  ▭  Coordinates      │
│  │ ID:1│            │  ID:1 X:10.5m Y:5.2m│
│  └─────┘            │  ID:2 X:15.3m Y:8.1m│
└─────────────────────┴──────────────────────┘
```

**LEFT Screen (Mắt thần AI):**

- Video gốc với khung xoay (OBB) của xe
- Hiển thị ID phương tiện
- FPS counter
- Số lượng detection

**RIGHT Screen (Bản sao số BEV):**

- Nền bản đồ vệ tinh
- ● Chấm tròn (xanh lá) = Xe máy
- ▭ Hình chữ nhật (xanh dương) = Ô tô
- Tọa độ thực tế (mét) của các xe
- FPS counter
- Số lượng xe đang tracking

## 🔧 Cấu hình

### main.py

```python
WIN_W = 750           # Chiều rộng mỗi screen
WIN_H = 650           # Chiều cao mỗi screen

VIDEO_PATH = "data/myDemoVideo.mp4"
SATELLITE_PATH = "data/anh_ve_tinh_BEV.jpg"
STATIC_IMAGE_PATH = "data/anh_ngatu_tinh.jpg"
HOMOGRAPHY_PATH = "modules/homography.npy"
```

### detector.py

```python
Detector(
    model_path="weights/yolov8_obb.pt",
    conf_threshold=0.5,      # Ngưỡng confidence
    device="cpu"             # "cpu" hoặc "cuda"
)
```

### tracker.py

```python
Tracker(
    max_age=30,              # Số frame không thấy trước khi xóa track
    min_hits=3,              # Số detection cần để xác nhận track
    iou_threshold=0.3        # Ngưỡng matching
)
```

## 📊 Quy trình xử lý

```
Frame Video
    ↓
[DETECTOR] YOLOv8-OBB Detection
    ↓ (Detections + OBB)
[TRACKER] BoT-SORT Tracking
    ↓ (Vehicles + ID)
    ├──→ [Camera Screen] Draw OBB + ID
    └──→ [HOMOGRAPHY] Pixel → BEV
         ↓
         [BEV_MAPPER] Draw Symbols
         ↓
         [BEV Screen] Display on Satellite
```

## 🎯 Các thành phần chính

### 1. **detector.py** (HẠO)

- Tích hợp YOLOv8-OBB model
- Phát hiện xe máy + ô tô
- Auto-download weights
- Vẽ OBB boxes

**Đầu ra:** Danh sách detections

```python
{
    "u": 450.5,
    "v": 520.3,
    "cx": 450.5,
    "cy": 520.3,
    "obb_points": [[...], [...], [...], [...]],  # 4 điểm
    "confidence": 0.92,
    "class_id": 0,
    "class_name": "car",
    "angle": 15.3
}
```

### 2. **tracker.py** (KIM)

- BoT-SORT algorithm
- Kalman filter cho dự đoán chuyển động
- Giữ vững ID qua occlusion
- Xử lý "re-identification"

**Đầu ra:** Vehicles với ID

```python
{
    "id": 1,
    "class": "car",
    "u": 450.5,
    "v": 520.3,
    "confidence": 0.92,
    "age": 15,
    "hits": 15
}
```

### 3. **homography.py** (KIỆT - Phần 1)

- Calibrate 4 điểm trên mặt đường
- Tính ma trận homography
- Lưu/load ma trận

### 4. **bev_mapper.py** (KIỆT - Phần 2)

- Ánh xạ pixel camera → tọa độ BEV
- Vẽ biểu tượng xe trên bản đồ
- Tính toán tọa độ mét

### 5. **video_stream.py** (KHÔI)

- Multi-threading video reader
- Đọc mượt mà không giật lag

### 6. **main.py** (KHÔI)

- Orchestration toàn bộ hệ thống
- Tạo giao diện 2 màn hình
- Quy trình xử lý chính

## ⌨️ Phím tắt

| Phím        | Tác dụng           |
| ----------- | ------------------ |
| `ESC`       | Thoát chương trình |
| `Q`         | Thoát chương trình |
| Đóng cửa sổ | Thoát chương trình |

## 🐛 Troubleshooting

### 1. "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### 2. "No video found"

- Đặt file video vào thư mục `data/`
- Tên file: `myDemoVideo.mp4` hoặc thay đổi `VIDEO_PATH` trong main.py

### 3. "CUDA not available"

- Nếu không có GPU, để `device="cpu"` trong detector.py
- Xử lý sẽ chậm hơn nhưng vẫn hoạt động

### 4. "Model download failed"

```bash
# Download thủ công
yolo detect download yolov8x-obb.pt
```

### 5. Detector chạy quá chậm

- Giảm resolution video trong main.py
- Hoặc dùng model nhẹ hơn: `yolov8s-obb.pt` thay vì `yolov8x-obb.pt`

## 📈 Performance Tips

1. **GPU Acceleration:**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Sau đó dùng `device="cuda"` trong detector

2. **Video resolution:**

   ```python
   frame_small = cv2.resize(frame, (640, 480))  # Giảm độ phân giải
   ```

3. **Skip frames:**
   ```python
   if frame_count % 2 == 0:  # Process mỗi 2 frame
       detections = detector.detect(frame)
   ```

## 📝 Ghi chú phát triển

- Model YOLOv8-OBB sẽ tự động download lần đầu (~250MB)
- Ma trận homography được lưu vào `modules/homography.npy`
- Tracker reset mỗi lần chạy video mới

## 👥 Phân công

| TV   | Phần                           | File                         |
| ---- | ------------------------------ | ---------------------------- |
| KHÔI | Bước 1: Video + UI + Main      | main.py, video_stream.py     |
| HẠO  | Bước 3A: Detection             | detector.py                  |
| KIM  | Bước 3B: Tracking              | tracker.py                   |
| KIỆT | Bước 2, 4, 5: Homography + BEV | homography.py, bev_mapper.py |

## 📚 Tài liệu tham khảo

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)

## 📞 Support

Để debug hoặc cải thiện:

1. Kiểm tra console output
2. Xóa cache (nếu có vấn đề model)
3. Thử với video khác
4. Adjust ngưỡng confidence, max_age, min_hits

---

**Created:** 2026-08-14  
**Status:** Ready to use ✅
