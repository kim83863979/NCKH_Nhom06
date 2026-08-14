import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import cv2


class KalmanTracker:
    """
    Bộ lọc Kalman đơn giản để dự đoán vị trí trong frame tiếp theo.
    """

    def __init__(self, x, y, vx=0, vy=0, dt=0.033):
        """
        Args:
            x, y: Vị trí ban đầu
            vx, vy: Vận tốc ban đầu
            dt: Thời gian giữa các frame (mặc định ~30fps)
        """

        # Trạng thái: [x, y, vx, vy]
        self.state = np.array(
            [x, y, vx, vy],
            dtype=np.float32
        )

        # Ma trận chuyển tiếp trạng thái
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Nhiễu quá trình
        self.Q = np.eye(4) * 0.01

        # Nhiễu đo lường
        self.R = np.eye(2) * 10

        # Độ không chắc chắn hiện tại
        self.P = np.eye(4) * 10

    def predict(self):
        """Dự đoán vị trí tiếp theo."""

        self.state = self.F @ self.state

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[0], self.state[1]

    def update(self, x, y):
        """
        Cập nhật với phép đo mới.

        Args:
            x, y: Vị trí đo được từ detection
        """

        z = np.array([x, y])

        H = np.zeros((2, 4))
        H[0, 0] = 1
        H[1, 1] = 1

        y = z - (H @ self.state[:2])

        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = (
            self.state
            + K @ np.concatenate(
                [[y[0]], [y[1]], [0], [0]]
            )
        )

        self.P = (
            (np.eye(4) - K @ H)
            @ self.P
        )

        # Cập nhật vận tốc từ frame trước
        if hasattr(self, "prev_x"):

            self.state[2] = (
                x - self.prev_x
            )

            self.state[3] = (
                y - self.prev_y
            )

        self.prev_x = x
        self.prev_y = y


class Track:
    """
    Đối tượng theo dõi một chiếc xe qua các frame.
    """

    _id_counter = 1

    def __init__(self, detection, frame_idx):
        """
        Args:
            detection: Dict từ detector
            frame_idx: Số frame hiện tại
        """

        self.id = Track._id_counter
        Track._id_counter += 1

        self.detections = [detection]

        self.kalman = KalmanTracker(
            detection["u"],
            detection["v"]
        )

        self.last_seen = frame_idx

        self.age = 1

        self.hits = 1

        self.class_name = (
            detection["class_name"]
        )

    def predict(self):
        """Dự đoán vị trí tiếp theo."""

        x, y = self.kalman.predict()

        return {
            "u": x,
            "v": y,
            "id": self.id,
            "class_name": self.class_name
        }

    def update(self, detection, frame_idx):
        """
        Cập nhật track với detection mới.

        Args:
            detection: Dict từ detector
            frame_idx: Số frame hiện tại
        """

        self.detections.append(detection)

        self.kalman.update(
            detection["u"],
            detection["v"]
        )

        self.last_seen = frame_idx

        self.age += 1

        self.hits += 1

    def get_current(self):
        """Lấy detection cuối cùng."""

        if self.detections:
            return self.detections[-1]

        return None

    def is_tentative(self):
        """Track còn quá non, chưa đủ độ tin cậy."""

        return self.hits < 3

    def is_confirmed(self):
        """Track đã xác nhận, đủ tin cậy."""

        return self.hits >= 3

    def is_lost(self, max_age=30):
        """
        Track đã bị mất.

        Args:
            max_age: Số frame tối đa không thấy track
        """

        return self.age - self.last_seen > max_age


class Tracker:
    """
    Tracker sử dụng BoT-SORT-lite algorithm.

    Quá trình:
    1. Nhận detections từ detector
    2. Dự đoán vị trí các track hiện có
    3. Matching detections với tracks dựa trên khoảng cách
    4. Tạo track mới cho detection không match
    5. Xóa track bị mất
    6. Trả về danh sách tracks được xác nhận
    """

    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    ):
        """
        Args:
            max_age: Số frame tối đa không thấy track
            min_hits: Số detection tối thiểu để xác nhận track
            iou_threshold: Ngưỡng khoảng cách để matching
        """

        self.tracks = []

        self.max_age = max_age

        self.min_hits = min_hits

        self.iou_threshold = iou_threshold

        self.frame_count = 0

    # ==========================================================
    # TÍNH KHOẢNG CÁCH GIỮA 2 VỊ TRÍ
    # ==========================================================
    def _distance(
        self,
        track_pos,
        det_pos
    ):
        """
        Tính khoảng cách Euclidean giữa track và detection.

        Args:
            track_pos: (x, y) của track
            det_pos: (x, y) của detection

        Returns:
            Khoảng cách (float)
        """

        return np.sqrt(
            (track_pos[0] - det_pos[0]) ** 2
            + (track_pos[1] - det_pos[1]) ** 2
        )

    # ==========================================================
    # MATCHING DETECTIONS VỚI TRACKS
    # ==========================================================
    def _match_detections(
        self,
        detections
    ):
        """
        Ghép detections với tracks.

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """

        if not self.tracks or not detections:

            return (
                [],
                list(range(len(detections))),
                list(range(len(self.tracks)))
            )

        # --------------------------------------------------------
        # Tính ma trận khoảng cách
        # --------------------------------------------------------
        track_positions = []

        for track in self.tracks:

            pred = track.predict()

            track_positions.append(
                (pred["u"], pred["v"])
            )

        det_positions = [
            (det["u"], det["v"])
            for det in detections
        ]

        # Tính ma trận khoảng cách
        dist_matrix = cdist(
            track_positions,
            det_positions,
            metric="euclidean"
        )

        # --------------------------------------------------------
        # Matching tham lam
        # --------------------------------------------------------
        matched_pairs = []

        used_dets = set()
        used_tracks = set()

        # Sắp xếp theo khoảng cách nhỏ nhất
        flat_indices = np.argsort(
            dist_matrix.ravel()
        )

        for flat_idx in flat_indices:

            t_idx = flat_idx // len(detections)
            d_idx = flat_idx % len(detections)

            if (
                t_idx in used_tracks
                or d_idx in used_dets
            ):
                continue

            distance = dist_matrix[
                t_idx,
                d_idx
            ]

            # Ngưỡng matching
            max_distance = 100

            if distance < max_distance:

                matched_pairs.append(
                    (t_idx, d_idx)
                )

                used_tracks.add(t_idx)
                used_dets.add(d_idx)

        unmatched_dets = [
            i for i in range(len(detections))
            if i not in used_dets
        ]

        unmatched_tracks = [
            i for i in range(len(self.tracks))
            if i not in used_tracks
        ]

        return (
            matched_pairs,
            unmatched_dets,
            unmatched_tracks
        )

    # ==========================================================
    # CẬP NHẬT TRACKER
    # ==========================================================
    def update(self, detections):
        """
        Cập nhật tracker với detections mới.

        Args:
            detections: Danh sách dict từ detector

        Returns:
            Danh sách tracks được xác nhận (kèm ID)
        """

        self.frame_count += 1

        # ========================================================
        # 1. MATCHING
        # ========================================================
        matched_pairs, unmatched_dets, unmatched_tracks = (
            self._match_detections(detections)
        )

        # ========================================================
        # 2. CẬP NHẬT TRACKS ĐÃ MATCH
        # ========================================================
        for t_idx, d_idx in matched_pairs:

            self.tracks[t_idx].update(
                detections[d_idx],
                self.frame_count
            )

        # ========================================================
        # 3. TẠO TRACK MỚI CHO DETECTION KHÔNG MATCH
        # ========================================================
        for d_idx in unmatched_dets:

            track = Track(
                detections[d_idx],
                self.frame_count
            )

            self.tracks.append(track)

        # ========================================================
        # 4. XÓA TRACKS BỊ MẤT
        # ========================================================
        self.tracks = [
            track for track in self.tracks
            if not track.is_lost(self.max_age)
        ]

        # ========================================================
        # 5. LỌCTRACK ĐỦ TIN CẬY
        # ========================================================
        confirmed_tracks = [
            track for track in self.tracks
            if track.is_confirmed()
        ]

        # ========================================================
        # 6. RETURN KẾT QUẢ
        # ========================================================
        vehicles = []

        for track in confirmed_tracks:

            detection = track.get_current()

            if detection is None:
                continue

            vehicle = {
                "id": track.id,
                "class": detection[
                    "class_name"
                ],
                "u": detection["u"],
                "v": detection["v"],
                "confidence": detection[
                    "confidence"
                ],
                "age": track.age,
                "hits": track.hits
            }

            vehicles.append(vehicle)

        return vehicles

    # ==========================================================
    # RESET TRACKER
    # ==========================================================
    def reset(self):
        """Reset tracker cho video mới."""

        self.tracks = []

        self.frame_count = 0

        Track._id_counter = 1


# ==============================================================
# TEST ĐỘC LẬP
# ==============================================================
if __name__ == "__main__":

    tracker = Tracker(
        max_age=30,
        min_hits=3
    )

    # Giả lập detections
    test_detections = [
        {"u": 100, "v": 100, "class_name": "Car", "confidence": 0.9},
        {"u": 300, "v": 200, "class_name": "Motorbike", "confidence": 0.85},
    ]

    print("[TEST] Tracker")

    for frame_idx in range(10):

        print(f"\nFrame {frame_idx}")

        # Giả lập chuyển động
        test_detections[0]["u"] += 5
        test_detections[0]["v"] += 3

        test_detections[1]["u"] -= 4
        test_detections[1]["v"] += 2

        # Update
        vehicles = tracker.update(
            test_detections
        )

        print(
            f"  Vehicles: {len(vehicles)}"
        )

        for v in vehicles:

            print(
                f"    - ID {v['id']}: "
                f"{v['class']} at "
                f"({v['u']:.0f}, {v['v']:.0f})"
            )

    print("\n[TEST] Hoan thanh")
