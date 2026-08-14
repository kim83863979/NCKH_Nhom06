"""
Giải pháp Auto-OBB từ Model YOLO 2D Thường (Standard Axis-Aligned Bounding Box).
--------------------------------------------------------------------------------
Tính năng: Tự động tính & tự điều chỉnh góc xoay (Auto Angle Adjustment) dựa trên
kết hợp Motion Vector + Computer Vision Image Analysis (MinAreaRect).

Input : Frame thô từ TV1 (BGR numpy array).
Output: Dữ liệu OBB 4 điểm xoay + Class xe -> Giao TV3.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
import math
import json
import os
from datetime import datetime
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Chưa cài ultralytics. Chạy: pip install ultralytics") from e


# ============================================================================
# CẤU HÌNH MACRO
# ============================================================================
# Dùng model YOLO chuẩn (không phải -obb)
MODEL_PATH = "yolo11s.pt"  # hoặc "yolo11s.pt" nếu xài Ultralytics mới

# 2. Hạ ngưỡng Confidence xuống 0.25 để giữ lại các detection xe máy
CONF_THRESHOLD = 0.25

# 3. Thêm Class ID 1 (bicycle) vì YOLO rất hay bị nhầm xe máy thành xe đạp
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]  # 1:


# ============================================================================
# DỮ LIỆU BÀN GIAO CHO TV3 (GIỮ NGUYÊN INTERFACE)
# ============================================================================
@dataclass
class OBBDetection:
    polygon: List[List[float]]  # 4 điểm [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    angle_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polygon": [[round(c, 2) for c in pt] for pt in self.polygon],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "track_id": self.track_id,
            "angle_deg": round(float(self.angle_deg), 2) if self.angle_deg is not None else 0.0
        }


@dataclass
class FrameOBBResult:
    frame_index: int
    detections: List[OBBDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "detections": [d.to_dict() for d in self.detections],
        }


# ============================================================================
# BỘ TỰ ĐỘNG ĐIỀU CHỈNH GÓC (AUTO-ANGLE ESTIMATOR)
# ============================================================================
class AutoAngleAdjuster:
    """
    Tự động tính toán & làm mượt góc xoay cho từng xe:
    - Nếu xe di chuyển: Dùng Vector hướng đi (Độ chính xác cao nhất).
    - Nếu xe đứng yên: Crop ảnh xe -> Phân tích cạnh (Canny + minAreaRect) tìm dáng xe.
    - Làm mượt góc qua thời gian bằng bộ lọc EMA (tránh giật khung).
    """
    def __init__(self, history_len: int = 6, min_move_px: float = 3.5, smoothing: float = 0.25):
        self.history_len = history_len
        self.min_move_px = min_move_px
        self.smoothing = smoothing
        
        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        self.smoothed_angles: Dict[int, float] = {}

    @staticmethod
    def _shortest_angle_diff(from_deg: float, to_deg: float) -> float:
        """Xử lý góc xoay qua mốc 180/-180 độ."""
        return (to_deg - from_deg + 180.0) % 360.0 - 180.0

    def _estimate_angle_from_image_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Fallback khi xe đứng yên: Crop vùng ảnh xe -> dùng Thuật toán xử lý ảnh 
        để tự ước lượng trục dài của thân xe.
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]
        
        # Clip tọa độ hợp lệ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return 0.0

        # Chuyển xám & lọc cạnh Canny
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Tìm Contour lớn nhất đại diện cho xe
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 20:
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                # Chuẩn hóa góc rect về khoảng [-90, 90]
                if rect[1][0] < rect[1][1]:
                    angle += 90.0
                return angle
        return 0.0

    def get_auto_angle(self, track_id: int, cx: float, cy: float, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        hist = self.track_history[track_id]
        hist.append((cx, cy))
        
        prev_angle = self.smoothed_angles.get(track_id, 0.0)
        target_angle = prev_angle

        # 1. Thử tính góc từ chuyển động (Motion Vector)
        if len(hist) >= 2:
            dx = hist[-1][0] - hist[0][0]
            dy = hist[-1][1] - hist[0][1]
            dist = math.hypot(dx, dy)
            
            if dist >= self.min_move_px:
                target_angle = math.degrees(math.atan2(dy, dx))
            else:
                # 2. Xe đứng yên -> Nếu chưa từng có góc thì dùng Image Crop Analysis
                if track_id not in self.smoothed_angles:
                    target_angle = self._estimate_angle_from_image_crop(frame, bbox)

        # 3. Làm mượt góc (EMA Filter)
        delta = self._shortest_angle_diff(prev_angle, target_angle)
        new_angle = prev_angle + self.smoothing * delta
        
        self.smoothed_angles[track_id] = new_angle
        return new_angle


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================
class StandardYoloAutoOBB:
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONF_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.angle_adjuster = AutoAngleAdjuster()

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameOBBResult:
        # Bật tracking tự động của YOLO
        results = self.model.track(source=frame, conf=self.conf_threshold, persist=True, verbose=False)
        detections: List[OBBDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            # Lấy thông tin BBox 2D chuẩn
            boxes_xywh = r.boxes.xywh.cpu().numpy()
            boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            
            # Track ID (nếu chưa có track_id thì fallback gán tạm ID=-1)
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(class_ids)
            names = r.names

            for (cx, cy, w, h), bbox_xyxy, tid, cls_id, conf in zip(boxes_xywh, boxes_xyxy, track_ids, class_ids, confs):
                # Chỉ lọc phương tiện giao thông
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                class_name = names.get(int(cls_id), str(cls_id))

                # TỰ ĐỘNG BẮT GÓC XOAY
                angle = self.angle_adjuster.get_auto_angle(
                    track_id=int(tid),
                    cx=float(cx), cy=float(cy),
                    frame=frame, bbox=tuple(bbox_xyxy)
                )

                # DỰNG 4 ĐIỂM XOAY POLYGON TỪ GÓC MỚI TÍNH
                polygon = self._build_obb_polygon(cx, cy, w, h, angle)

                detections.append(OBBDetection(
                    polygon=polygon,
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    track_id=int(tid),
                    angle_deg=angle
                ))

        return FrameOBBResult(frame_index=frame_index, detections=detections)

    @staticmethod
    def _build_obb_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[List[float]]:
        """Chuyển (cx, cy, w, h, angle) -> [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"""
        length, width = max(w, h), min(w, h)
        half_l, half_w = length / 2.0, width / 2.0

        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated = np.dot(corners, rot_mat.T) + np.array([cx, cy])
        return rotated.tolist()


# ============================================================================
# PIPELINE XỬ LÝ VIDEO
# ============================================================================
def run_auto_obb_demo(
    video_source: Any = "/kaggle/input/datasets/holthin/testvideo/32499-392669624_medium.mp4",
    output_video_path: str = "/kaggle/working/output_auto_obb.mp4",
    log_output_path: Optional[str] = "/kaggle/working/detection_log.json",
    max_frames: Optional[int] = None
):
    detector = StandardYoloAutoOBB()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_results_for_tv3 = []
    frame_index = 0

    print("🚀 Đang chạy Auto-Adjustable OBB với Model YOLO Thường...")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames is not None and frame_index >= max_frames):
                break

            # 1. Chạy Detector
            res = detector.process_frame(frame, frame_index=frame_index)
            
            # 2. Dữ liệu giao TV3
            output_data = res.to_dict()
            all_results_for_tv3.append(output_data)

            # 3. Vẽ kết quả lên video
            for det in res.detections:
                pts = np.array(det.polygon, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                x, y = pts[0]
                cv2.putText(
                    frame, 
                    f"{det.class_name} #{det.track_id} {det.angle_deg:.0f} deg", 
                    (x, max(y - 5, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
                )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()
        print(f"✅ Hoàn tất! Video lưu tại: {output_video_path}")

    # ---- GHI LOG FILE ----
    if log_output_path:
        log_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "video_source": str(video_source),
                "model": MODEL_PATH,
                "conf_threshold": CONF_THRESHOLD,
                "total_frames": frame_index,
                "total_detections": sum(len(f["detections"]) for f in all_results_for_tv3),
            },
            "frames": all_results_for_tv3
        }

        os.makedirs(os.path.dirname(log_output_path) or ".", exist_ok=True)
        with open(log_output_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        file_size_mb = os.path.getsize(log_output_path) / (1024 * 1024)
        print(f"📄 Log đã lưu: {log_output_path} ({file_size_mb:.1f} MB)")

    return all_results_for_tv3


if __name__ == "__main__":
    results = run_auto_obb_demo(
        video_source="/kaggle/input/datasets/holthin/stream-video/2026-08-14 16-53-48.mp4",
        output_video_path="/kaggle/working/output_auto_obb.mp4",
        max_frames=None
    )