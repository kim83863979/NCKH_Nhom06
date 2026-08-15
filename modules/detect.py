"""
Giải pháp Auto-OBB từ Model YOLO 2D Thường (Standard Axis-Aligned Bounding Box).
--------------------------------------------------------------------------------
Tính năng: Tự động tính & tự điều chỉnh góc xoay + Angle Lock chống xoay khi đứng yên.

BẢN TỐI ƯU:
- Dọn track "chết" định kỳ -> chống memory leak khi chạy video dài.
- Ghi log JSON dạng streaming (JSONL, ghi ngay mỗi frame) -> không giữ cả video trong RAM,
  không mất dữ liệu nếu crash giữa chừng.
- VEHICLE_CLASS_IDS chuyển sang set -> lookup O(1).
- Chỉ định device + half precision (nếu có GPU) -> tăng tốc inference.
- Đọc/ghi video theo mô hình producer-consumer (threading) -> I/O không chặn GPU.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
import math
import json
import os
import threading
import queue
from datetime import datetime
import cv2
import numpy as np
import torch

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
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # set thay vì list -> lookup O(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"  # half precision chỉ có ý nghĩa trên GPU

# Track "chết" nếu không xuất hiện quá số frame này -> bị dọn khỏi bộ nhớ
TRACK_STALE_AFTER_FRAMES = 90  # ~3-4s ở 25-30fps, chỉnh theo fps thực tế
TRACK_CLEANUP_INTERVAL = 60  # dọn dẹp mỗi N frame, không cần check mỗi frame


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
            "angle_deg": round(float(self.angle_deg), 2) if self.angle_deg is not None else 0.0,
        }


@dataclass
class FrameOBBResult:
    frame_index: int
    detections: List[OBBDetection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_index": self.frame_index, "detections": [d.to_dict() for d in self.detections]}


# ============================================================================
# BỘ TỰ ĐỘNG ĐIỀU CHỈNH & KHÓA GÓC XOAY (STABLE AUTO-ANGLE ADJUSTER)
# ============================================================================
class AutoAngleAdjuster:
    """
    Tự động tính toán, làm mượt & ĐÓNG BẰNG góc xoay khi xe đứng yên:
    - Di chuyển (> 12px): Cập nhật góc theo Motion Vector + EMA Smoothing.
    - Đứng yên (<= 12px): Khóa cứng góc (Angle Lock), không recalculate.
    - Track không xuất hiện lâu -> bị dọn khỏi bộ nhớ (chống leak).
    """

    def __init__(self, history_len: int = 8, min_move_px: float = 12.0, smoothing: float = 0.20):
        self.history_len = history_len
        self.min_move_px = min_move_px
        self.smoothing = smoothing

        self.track_history = defaultdict(lambda: deque(maxlen=history_len))
        self.smoothed_angles: Dict[int, float] = {}
        # frame_index cuối cùng mỗi track_id được cập nhật -> dùng để dọn track chết
        self.last_seen_frame: Dict[int, int] = {}

    @staticmethod
    def _shortest_angle_diff(from_deg: float, to_deg: float) -> float:
        """Xử lý góc xoay qua mốc 180/-180 độ."""
        return (to_deg - from_deg + 180.0) % 360.0 - 180.0

    def _estimate_initial_stationary_angle(self, w: float, h: float) -> float:
        """
        Khởi tạo góc ban đầu cho xe vừa xuất hiện nhưng ĐÃ ĐỨNG YÊN:
        Dựa vào tỷ lệ W/H của BBox thay vì soi Canny bị nhiễu.
        Lưu ý: chỉ là ước lượng thô ban đầu, sẽ tự sửa khi xe di chuyển.
        """
        return 0.0 if w >= h else 90.0

    def get_auto_angle(self, track_id: int, cx: float, cy: float, w: float, h: float, frame_index: int = 0) -> float:
        hist = self.track_history[track_id]
        hist.append((cx, cy))
        self.last_seen_frame[track_id] = frame_index

        if track_id not in self.smoothed_angles:
            initial_angle = self._estimate_initial_stationary_angle(w, h)
            self.smoothed_angles[track_id] = initial_angle

        prev_angle = self.smoothed_angles[track_id]

        if len(hist) >= 3:
            dx = hist[-1][0] - hist[0][0]
            dy = hist[-1][1] - hist[0][1]
            dist = math.hypot(dx, dy)

            if dist >= self.min_move_px:
                target_angle = math.degrees(math.atan2(dy, dx))
                delta = self._shortest_angle_diff(prev_angle, target_angle)
                new_angle = prev_angle + self.smoothing * delta
                self.smoothed_angles[track_id] = new_angle
                return new_angle

        return prev_angle

    def cleanup_stale_tracks(self, current_frame: int, stale_after: int = TRACK_STALE_AFTER_FRAMES) -> int:
        """Xóa các track không xuất hiện quá `stale_after` frame -> chống memory leak."""
        stale_ids = [
            tid for tid, last_frame in self.last_seen_frame.items() if current_frame - last_frame > stale_after
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.smoothed_angles.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
        return len(stale_ids)


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================
class StandardYoloAutoOBB:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = CONF_THRESHOLD,
        imgsz: int = IMAGE_SIZE,
        device: str = DEVICE,
        half: bool = USE_HALF,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.angle_adjuster = AutoAngleAdjuster()

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameOBBResult:
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            persist=True,
            verbose=False,
            device=self.device,
            half=self.half,
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

                angle = self.angle_adjuster.get_auto_angle(
                    track_id=int(tid), cx=float(cx), cy=float(cy), w=float(w), h=float(h), frame_index=frame_index
                )

                polygon = self._build_obb_polygon(cx, cy, w, h, angle)

                detections.append(
                    OBBDetection(
                        polygon=polygon,
                        class_id=int(cls_id),
                        class_name=class_name,
                        confidence=float(conf),
                        track_id=int(tid),
                        angle_deg=angle,
                    )
                )

        # Dọn track chết định kỳ, không cần check mỗi frame để đỡ overhead
        if frame_index % TRACK_CLEANUP_INTERVAL == 0:
            self.angle_adjuster.cleanup_stale_tracks(frame_index)

        return FrameOBBResult(frame_index=frame_index, detections=detections)

    @staticmethod
    def _build_obb_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[List[float]]:
        """Chuyển (cx, cy, w, h, angle) -> [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"""
        length, width = max(w, h), min(w, h)
        half_l, half_w = length / 2.0, width / 2.0

        corners = np.array([[-half_l, -half_w],
                            [half_l, -half_w],
                            [half_l, half_w],
                            [-half_l, half_w]])

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot_mat = np.array([[cos_a, -sin_a],
                            [sin_a, cos_a]])

        rotated = np.dot(corners, rot_mat.T) + np.array([cx, cy])
        return rotated.tolist()


# ============================================================================
# PIPELINE XỬ LÝ VIDEO (đọc frame trên thread riêng để không chặn GPU)
# ============================================================================
def _frame_reader(cap: cv2.VideoCapture, frame_q: "queue.Queue", max_frames: Optional[int]):
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_q.put((idx, frame))
        idx += 1
    frame_q.put(None)  # tín hiệu kết thúc


def run_auto_obb_demo(
    video_source: Any = "/kaggle/input/datasets/holthin/stream-video/2026-08-14 16-53-48.mp4",
    output_video_path: str = "/kaggle/working/output_auto_obb.mp4",
    log_output_path: Optional[str] = "/kaggle/working/detection_log.jsonl",
    max_frames: Optional[int] = None,
    reader_queue_size: int = 8,
):
    detector = StandardYoloAutoOBB()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Đọc frame trên thread riêng -> GPU không phải chờ disk I/O / decode
    frame_q: "queue.Queue" = queue.Queue(maxsize=reader_queue_size)
    reader_thread = threading.Thread(target=_frame_reader, args=(cap, frame_q, max_frames), daemon=True)
    reader_thread.start()

    total_detections = 0
    frame_index = 0

    print(
        f"🚀 Đang chạy Auto-Adjustable OBB trên '{DEVICE}' (half={USE_HALF}), Angle Lock bật, "
        f"streaming log + auto cleanup track..."
    )

    log_file = None
    if log_output_path:
        os.makedirs(os.path.dirname(log_output_path) or ".", exist_ok=True)
        log_file = open(log_output_path, "w", encoding="utf-8")

    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
            frame_index, frame = item

            res = detector.process_frame(frame, frame_index=frame_index)
            total_detections += len(res.detections)

            # Ghi JSONL: mỗi frame 1 dòng JSON -> streaming, không giữ cả video trong RAM
            if log_file:
                log_file.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")

            for det in res.detections:
                pts = np.array(det.polygon, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                x, y = pts[0]
                cv2.putText(
                    frame,
                    f"{det.class_name} #{det.track_id} {det.angle_deg:.0f} deg",
                    (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            writer.write(frame)

    finally:
        cap.release()
        writer.release()
        if log_file:
            log_file.close()
        print(f"✅ Hoàn tất! Video lưu tại: {output_video_path}")
        if log_output_path:
            file_size_mb = os.path.getsize(log_output_path) / (1024 * 1024)
            print(
                f"📄 Log JSONL đã lưu: {log_output_path} ({file_size_mb:.1f} MB), "
                f"{frame_index + 1} frame, {total_detections} detection."
            )

    return {
        "total_frames": frame_index + 1,
        "total_detections": total_detections,
        "log_path": log_output_path,
        "video_path": output_video_path,
    }


if __name__ == "__main__":
    summary = run_auto_obb_demo(
        video_source="/kaggle/input/datasets/holthin/stream-video/2026-08-14 16-53-48.mp4",
        output_video_path="/kaggle/working/output_auto_obb.mp4",
        max_frames=None,
    )
    print(summary)
