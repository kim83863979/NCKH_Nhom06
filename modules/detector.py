import cv2
import numpy as np
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "[WARNING] ultralytics chua duoc cai dat."
    )
    print(
        "Chay: pip install ultralytics"
    )
    exit()


class Detector:
    """
    Phát hiện và khoanh hình xe bằng YOLOv8-OBB.
    
    Đầu vào: Frame video
    Đầu ra: Danh sách các khung OBB kèm class và confidence
    """

    def __init__(
        self,
        model_path="weights/yolov8_obb.pt",
        conf_threshold=0.5,
        device="cpu"
    ):
        """
        Args:
            model_path: Đường dẫn đến file weights YOLOv8-OBB
            conf_threshold: Ngưỡng confidence tối thiểu
            device: "cpu" hoặc "cuda"
        """

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None

        self._load_model()

    # ==========================================================
    # LOAD MÔ HÌNH
    # ==========================================================
    def _load_model(self):
        """
        Tải mô hình YOLOv8-OBB.
        Nếu file không tồn tại, tự động download.
        """

        print()
        print("=" * 60)
        print(" YOLOV8-OBB DETECTOR")
        print("=" * 60)

        # --------------------------------------------------------
        # Kiểm tra file weights
        # --------------------------------------------------------
        if not os.path.exists(self.model_path):

            print()
            print(
                f"[INFO] Khong tim thay model: {self.model_path}"
            )

            print(
                "[INFO] Dang tai model tu Ultralytics..."
            )

            # Tạo folder weights nếu chưa có
            folder = os.path.dirname(
                self.model_path
            )

            if folder:
                os.makedirs(
                    folder,
                    exist_ok=True
                )

            # YOLOv8 sẽ tự động download
            # và lưu vào ~/.yolov8/

            try:
                self.model = YOLO(
                    "yolov8x-obb.pt"
                )

                # Copy model sang folder weights
                src = str(
                    Path.home() 
                    / ".cache" 
                    / "ultralytics" 
                    / "hub"
                    / "yolov8x-obb.pt"
                )

                if os.path.exists(src):
                    import shutil

                    shutil.copy(
                        src,
                        self.model_path
                    )

                    print(
                        f"[SUCCESS] Da luu model: "
                        f"{self.model_path}"
                    )

            except Exception as e:

                print(
                    f"[ERROR] Loi tai model: {e}"
                )

                print(
                    "[INFO] Su dung model tu YOLO cache"
                )

                self.model = YOLO(
                    "yolov8x-obb.pt"
                )

        else:

            print()
            print(
                f"[INFO] Load model: {self.model_path}"
            )

            self.model = YOLO(
                self.model_path
            )

        print(
            "[SUCCESS] Model da load xong."
        )

        print("=" * 60)
        print()

    # ==========================================================
    # PHÁT HIỆN XE
    # ==========================================================
    def detect(self, frame):
        """
        Phát hiện xe trong một frame.

        Args:
            frame: Hình ảnh numpy array (BGR)

        Returns:
            Danh sách dict chứa:
            - u, v: Tọa độ tâm
            - cx, cy: Tọa độ tâm (alias)
            - obb_points: 4 điểm của hộp xoay
            - confidence: Độ tin cậy
            - class_id: ID class
            - class_name: Tên class
            - angle: Góc xoay (độ)
        """

        if frame is None:
            return []

        # --------------------------------------------------------
        # INFERENCE
        # --------------------------------------------------------
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []

        for result in results:

            if result.obb is None:
                continue

            boxes_obb = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            cls_ids = result.obb.cls.cpu().numpy()

            for i, box in enumerate(boxes_obb):

                conf = float(confs[i])
                cls_id = int(cls_ids[i])

                # ------------------------------------------------
                # Lấy tên class
                # ------------------------------------------------
                class_name = (
                    self.model.names.get(
                        cls_id,
                        f"Class_{cls_id}"
                    )
                )

                # ------------------------------------------------
                # Tính tâm của hộp xoay
                # ------------------------------------------------
                # box = [x1, y1, x2, y2, x3, y3, x4, y4]
                # 4 điểm của hình chữ nhật xoay

                box_2d = box.reshape(4, 2)

                cx = np.mean(box_2d[:, 0])
                cy = np.mean(box_2d[:, 1])

                # ------------------------------------------------
                # Tính góc xoay
                # ------------------------------------------------
                dx = box_2d[1, 0] - box_2d[0, 0]
                dy = box_2d[1, 1] - box_2d[0, 1]

                angle = np.arctan2(
                    dy,
                    dx
                ) * 180 / np.pi

                # ------------------------------------------------
                # Tạo detection dict
                # ------------------------------------------------
                detection = {
                    "u": float(cx),
                    "v": float(cy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "obb_points": box_2d.astype(
                        np.float32
                    ),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": str(class_name),
                    "angle": float(angle)
                }

                detections.append(
                    detection
                )

        return detections

    # ==========================================================
    # VẼ DETECTION LÊN FRAME
    # ==========================================================
    def draw_detections(
        self,
        frame,
        detections,
        thickness=2,
        text_scale=0.6
    ):
        """
        Vẽ các khung OBB và thông tin lên frame.

        Args:
            frame: Hình ảnh gốc
            detections: Danh sách detection từ detect()
            thickness: Độ dày đường vẽ
            text_scale: Kích thước chữ

        Returns:
            Frame đã vẽ
        """

        frame_draw = frame.copy()

        for det in detections:

            # ------------------------------------------------
            # Màu sắc theo class
            # ------------------------------------------------
            class_name = det["class_name"].lower()

            if (
                "motor" in class_name
                or "bike" in class_name
                or "xe may" in class_name
                or "xe máy" in class_name
            ):
                color = (0, 255, 0)  # Xanh lá

            else:
                color = (255, 0, 0)  # Xanh dương (BGR)

            # ------------------------------------------------
            # Vẽ hộp OBB
            # ------------------------------------------------
            obb = det["obb_points"].astype(
                np.int32
            )

            cv2.polylines(
                frame_draw,
                [obb],
                True,
                color,
                thickness
            )

            # ------------------------------------------------
            # Vẽ tâm
            # ------------------------------------------------
            center = (
                int(det["u"]),
                int(det["v"])
            )

            cv2.circle(
                frame_draw,
                center,
                5,
                color,
                -1
            )

            # ------------------------------------------------
            # Vẽ thông tin
            # ------------------------------------------------
            text = (
                f"{det['class_name'][:6]} "
                f"({det['confidence']:.2f})"
            )

            text_pos = (
                int(det["u"]) - 50,
                int(det["v"]) - 20
            )

            cv2.putText(
                frame_draw,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                color,
                thickness
            )

        return frame_draw

    # ==========================================================
    # LỌC DETECTION THEO CLASS
    # ==========================================================
    def filter_by_class(
        self,
        detections,
        class_names=None
    ):
        """
        Lọc detection chỉ giữ lại những class cần.

        Args:
            detections: Danh sách detection
            class_names: Danh sách tên class cần giữ
                        (nếu None, giữ toàn bộ)

        Returns:
            Danh sách detection đã lọc
        """

        if class_names is None:
            return detections

        filtered = []

        for det in detections:

            class_name_lower = (
                str(
                    det["class_name"]
                ).lower()
            )

            for target_class in class_names:

                if (
                    target_class.lower()
                    in class_name_lower
                ):

                    filtered.append(det)
                    break

        return filtered


# ==============================================================
# TEST ĐỘC LẬP
# ==============================================================
if __name__ == "__main__":

    import glob

    # ============================================================
    # Tìm video
    # ============================================================
    video_path = None

    for ext in (
        "*.mp4",
        "*.avi",
        "*.mov",
        "*.mkv"
    ):

        videos = glob.glob(
            f"data/{ext}"
        )

        if videos:
            video_path = videos[0]
            break

    if video_path is None:

        print(
            "[ERROR] Khong tim thay video"
        )

        exit()

    print(f"[TEST] Video: {video_path}")

    # ============================================================
    # Tạo detector
    # ============================================================
    detector = Detector(
        model_path="weights/yolov8_obb.pt",
        conf_threshold=0.5,
        device="cpu"
    )

    # ============================================================
    # Đọc video test
    # ============================================================
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 5 != 0:
            continue

        print(
            f"\n[TEST] Frame {frame_count}"
        )

        # Resize để test nhanh
        frame_small = cv2.resize(
            frame,
            (640, 480)
        )

        # Detect
        detections = detector.detect(
            frame_small
        )

        print(
            f"  Phat hien: {len(detections)} xe"
        )

        for det in detections:

            print(
                f"    - {det['class_name']}: "
                f"({det['u']:.0f}, {det['v']:.0f}) "
                f"[{det['confidence']:.2f}]"
            )

        # Draw
        frame_out = detector.draw_detections(
            frame_small,
            detections
        )

        cv2.imshow(
            "YOLOv8-OBB Detection",
            frame_out
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break

        # Test chỉ 10 frame
        if frame_count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[TEST] Hoan thanh")
