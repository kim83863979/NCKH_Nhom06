"""
Giải pháp Auto-OBB từ Model YOLO 2D Thường (Standard Axis-Aligned Bounding Box).
--------------------------------------------------------------------------------
Tính năng: Tự động tính & tự điều chỉnh góc xoay + Angle Lock chống xoay khi đứng yên.
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
MODEL_PATH = "yolo11s.pt"  
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 1024  # Tăng resolution để bắt xe máy nhỏ xa
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]  # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck


# ============================================================================
# DỮ LIỆU BÀN GIAO CHO TV3
# ============================================================================
@dataclass
class OBBDetection:
    polygon: List[List[float]]
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
# BỘ TỰ ĐỘNG ĐIỀU CHỈNH & KHÓA GÓC XOAY (STABLE AUTO-ANGLE ADJUSTER)
# ============================================================================
class AutoAngleAdjuster:
    """
    Tự động tính toán, làm mượt & ĐÓNG BẰNG góc xoay khi xe đứng yên:
    - Di chuyển (> 12px): Cập nhật góc theo Motion Vector + EMA Smoothing.
    - Đứng yên (<= 12px): Khóa cứng góc (Angle Lock), không recalculate.
    """
    def __init__(self, history_len: int = 8, min_move_px: float = 12.0, smoothing: float = 0.20):
        self.history_len = history_len
        self.min_move_px = min_move_px  # Tăng ngưỡng để lọc hoàn toàn nhiễu BBox jitter
        self.smoothing = smoothing
        
        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        self.smoothed_angles: Dict[int, float] = {}

    @staticmethod
    def _shortest_angle_diff(from_deg: float, to_deg: float) -> float:
        """Xử lý góc xoay qua mốc 180/-180 độ."""
        return (to_deg - from_deg + 180.0) % 360.0 - 180.0

    def _estimate_initial_stationary_angle(self, w: float, h: float) -> float:
        """
        Khởi tạo góc ban đầu cho xe vừa xuất hiện nhưng ĐÃ ĐỨNG YÊN:
        Dựa vào tỷ lệ W/H của BBox thay vì soi Canny bị nhiễu.
        """
        return 0.0 if w >= h else 90.0

    def get_auto_angle(self, track_id: int, cx: float, cy: float, w: float, h: float) -> float:
        hist = self.track_history[track_id]
        hist.append((cx, cy))

        # 1. Nếu xe mới xuất hiện chưa từng có góc -> Khởi tạo góc tĩnh ban đầu
        if track_id not in self.smoothed_angles:
            initial_angle = self._estimate_initial_stationary_angle(w, h)
            self.smoothed_angles[track_id] = initial_angle

        prev_angle = self.smoothed_angles[track_id]

        # 2. Kiểm tra khoảng cách di chuyển trong chuỗi history_len frame
        if len(hist) >= 3:
            dx = hist[-1][0] - hist[0][0]
            dy = hist[-1][1] - hist[0][1]
            dist = math.hypot(dx, dy)

            if dist >= self.min_move_px:
                # XE ĐANG DI CHUYỂN RÕ RÀNG -> Cập nhật góc mới từ Motion Vector
                target_angle = math.degrees(math.atan2(dy, dx))
                
                # Làm mượt góc qua thời gian bằng bộ lọc EMA
                delta = self._shortest_angle_diff(prev_angle, target_angle)
                new_angle = prev_angle + self.smoothing * delta
                
                self.smoothed_angles[track_id] = new_angle
                return new_angle

        # 3. XE ĐỨNG YÊN (hoặc chưa đủ history) -> KHÓA GÓC HOÀN TOÀN (ANGLE LOCK)
        return prev_angle


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================
class StandardYoloAutoOBB:
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONF_THRESHOLD, imgsz: int = IMAGE_SIZE):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.angle_adjuster = AutoAngleAdjuster()

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameOBBResult:
        results = self.model.track(
            source=frame, 
            conf=self.conf_threshold, 
            imgsz=self.imgsz, 
            persist=True, 
            verbose=False
        )
        detections: List[OBBDetection] = []

        for r in results:
            if r.boxes is None:
                continue

            boxes_xywh = r.boxes.xywh.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(class_ids)
            names = r.names

            for (cx, cy, w, h), tid, cls_id, conf in zip(boxes_xywh, track_ids, class_ids, confs):
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                class_name = names.get(int(cls_id), str(cls_id))

                # LẤY GÓC TỰ ĐỘNG (ĐÃ CÓ CƠ CHẾ ANGLE LOCK CHỐNG XOAY CHI ĐỨNG YÊN)
                angle = self.angle_adjuster.get_auto_angle(
                    track_id=int(tid),
                    cx=float(cx), cy=float(cy),
                    w=float(w), h=float(h)
                )

                # DỰNG POLYGON 4 ĐIỂM
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
    video_source: Any = "/kaggle/input/datasets/holthin/stream-video/2026-08-14 16-53-48.mp4",
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

    print("🚀 Đang chạy Auto-Adjustable OBB (Đã bật Angle Lock chống xoay khi xe đứng yên)...")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or (max_frames is not None and frame_index >= max_frames):
                break

            res = detector.process_frame(frame, frame_index=frame_index)
            
            output_data = res.to_dict()
            all_results_for_tv3.append(output_data)

            # Vẽ kết quả
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

    # Ghi log JSON
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