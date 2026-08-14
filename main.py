import cv2
import numpy as np
import os
import glob
import time
import sys

# Đảm bảo chúng ta ở đúng thư mục (Fix path issue)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from utils.video_stream import VideoStream
from modules.homography import Homography
from modules.bev_mapper import BEVMapper
from modules.detector import Detector
from modules.tracker import Tracker


# ============================================================
# CẤU HÌNH GIAO DIỆN
# ============================================================

WIN_W = 750
WIN_H = 650

WIN_LEFT = "MAT THAN AI"
WIN_RIGHT = "BAN SAO SO VE TINH"

VIDEO_PATH = "data/myDemoVideo.mp4"
SATELLITE_PATH = "data/anh_ve_tinh_BEV.jpg"
STATIC_IMAGE_PATH = "data/anh_ngatu_tinh.jpg"

HOMOGRAPHY_PATH = "modules/homography.npy"


# ============================================================
# TÌM VIDEO
# ============================================================

def get_video_source():

    if os.path.exists(VIDEO_PATH):
        return VIDEO_PATH

    exts = (
        "*.mp4",
        "*.avi",
        "*.mkv",
        "*.mov"
    )

    files = []

    for ext in exts:
        files.extend(
            glob.glob(
                os.path.join(
                    "data",
                    ext
                )
            )
        )

    if len(files) > 0:
        return files[0]

    return 0


# ============================================================
# LOAD ẢNH VỆ TINH
# ============================================================

def load_satellite():

    if not os.path.exists(SATELLITE_PATH):

        print(
            "[WARNING] Khong tim thay anh ve tinh:"
        )

        print(SATELLITE_PATH)

        image = np.zeros(
            (WIN_H, WIN_W, 3),
            dtype=np.uint8
        )

        image[:] = (40, 40, 40)

        return image

    image = cv2.imread(
        SATELLITE_PATH
    )

    if image is None:

        print(
            "[ERROR] Khong doc duoc anh ve tinh."
        )

        return np.zeros(
            (WIN_H, WIN_W, 3),
            dtype=np.uint8
        )

    image = cv2.resize(
        image,
        (WIN_W, WIN_H)
    )

    return image


# ============================================================
# TẠO HOMOGRAPHY
# ============================================================

def setup_homography():

    homography = Homography()

    # --------------------------------------------------------
    # Nếu đã có ma trận thì load
    # --------------------------------------------------------

    if os.path.exists(HOMOGRAPHY_PATH):

        print()
        print(
            "[HOMOGRAPHY] Phat hien ma tran da luu."
        )

        choice = input(
            "Su dung ma tran hien tai? (Y/N): "
        ).strip().lower()

        if choice == "y":

            homography.load(
                HOMOGRAPHY_PATH
            )

            return homography

    # --------------------------------------------------------
    # Kiểm tra ảnh tĩnh
    # --------------------------------------------------------

    if not os.path.exists(
        STATIC_IMAGE_PATH
    ):

        print(
            "[ERROR] Khong tim thay anh:"
        )

        print(
            STATIC_IMAGE_PATH
        )

        return None

    image = cv2.imread(
        STATIC_IMAGE_PATH
    )

    if image is None:

        print(
            "[ERROR] Khong doc duoc anh."
        )

        return None

    # --------------------------------------------------------
    # CLICK 4 ĐIỂM
    # --------------------------------------------------------

    points = homography.select_points(
        image
    )

    if points is None:

        print(
            "[ERROR] Khong chon du 4 diem."
        )

        return None

    # --------------------------------------------------------
    # NHẬP KÍCH THƯỚC THỰC TẾ
    # --------------------------------------------------------

    # Đóng tất cả cửa sổ CV2 để lấy focus lại cho terminal
    cv2.destroyAllWindows()
    time.sleep(0.5)
    sys.stdout.flush()

    try:

        width = float(
            input(
                "\nNhap chieu rong thuc te (m): "
            )
        )

        height = float(
            input(
                "Nhap chieu dai thuc te (m): "
            )
        )

    except ValueError:

        print(
            "[ERROR] Kich thuoc khong hop le."
        )

        return None

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scale = 20

    homography.create_destination_points(
        width,
        height,
        scale
    )

    # --------------------------------------------------------
    # TÍNH MA TRẬN
    # --------------------------------------------------------

    homography.calculate()

    # --------------------------------------------------------
    # LƯU
    # --------------------------------------------------------

    homography.save(
        HOMOGRAPHY_PATH
    )

    return homography


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" TRAFFIC BEV PROJECT")
    print("=" * 60)

    # ========================================================
    # 1. KHỞI TẠO HOMOGRAPHY
    # ========================================================

    homography = setup_homography()

    if homography is None:

        print(
            "[ERROR] Khong khoi tao duoc Homography."
        )

        return

    # ========================================================
    # 2. TẠO BEV MAPPER
    # ========================================================

    # Với scale = 20 pixel / mét
    #
    # Destination points của Homography sẽ xác định
    # kích thước vùng BEV.

    dst_points = homography.dst_points

    max_x = int(
        np.max(dst_points[:, 0])
    )

    max_y = int(
        np.max(dst_points[:, 1])
    )

    bev_mapper = BEVMapper(
        homography_matrix=homography.matrix,
        bev_width=max_x,
        bev_height=max_y,
        scale=20
    )

    # ========================================================
    # 3. LOAD BẢN ĐỒ VỆ TINH
    # ========================================================

    satellite_base = load_satellite()

    # ========================================================
    # 4. LOAD VIDEO
    # ========================================================

    video_path = get_video_source()

    print()
    print(
        "[VIDEO]",
        video_path
    )

    vs = VideoStream(
        video_path
    ).start()

    time.sleep(
        0.5
    )

    # ========================================================
    # 4.5. KHỞI TẠO DETECTOR
    # ========================================================

    print()

    detector = Detector(
        model_path="weights/yolov8_obb.pt",
        conf_threshold=0.5,
        device="cpu"
    )

    # ========================================================
    # 4.6. KHỞI TẠO TRACKER
    # ========================================================

    print()

    tracker = Tracker(
        max_age=30,
        min_hits=3
    )

    print(
        "[TRACKER] Da khoi tao."
    )

    print()

    # ========================================================
    # 5. TẠO CỬA SỔ
    # ========================================================

    cv2.namedWindow(
        WIN_LEFT,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        WIN_RIGHT,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WIN_LEFT,
        WIN_W,
        WIN_H
    )

    cv2.resizeWindow(
        WIN_RIGHT,
        WIN_W,
        WIN_H
    )

    cv2.moveWindow(
        WIN_LEFT,
        50,
        80
    )

    cv2.moveWindow(
        WIN_RIGHT,
        50 + WIN_W + 15,
        80
    )

    # ========================================================
    # 6. VÒNG LẶP VIDEO
    # ========================================================

    fps_counter = 0
    fps_timer = time.time()

    while True:

        # ----------------------------------------------------
        # Đọc frame
        # ----------------------------------------------------

        frame_cam = vs.read()

        if frame_cam is None:

            time.sleep(
                0.01
            )

            continue

        # ====================================================
        # PHÁT HIỆN XE (DETECTION)
        # ====================================================

        detections = detector.detect(
            frame_cam
        )

        # Lọc chỉ giữ Xe máy + Ô tô
        detections = detector.filter_by_class(
            detections,
            class_names=[
                "car",
                "truck",
                "motorbike",
                "motorcycle",
                "xe may",
                "xe máy",
                "o to"
            ]
        )

        # ====================================================
        # THEO DÕI XE (TRACKING)
        # ====================================================

        vehicles = tracker.update(
            detections
        )

        # ====================================================
        # VẼ DETECTION + ID LÊN FRAME GỐC
        # ====================================================

        cam_display = detector.draw_detections(
            frame_cam,
            detections
        )

        # Vẽ ID và thông tin tracking
        for vehicle in vehicles:

            u = int(vehicle["u"])
            v = int(vehicle["v"])

            # Vẽ ID tracking
            cv2.putText(
                cam_display,
                f"ID:{vehicle['id']}",
                (u + 15, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Vẽ dấu hiệu
            cv2.circle(
                cam_display,
                (u, v),
                3,
                (0, 255, 255),
                -1
            )

        # Resize camera
        cam_display = cv2.resize(
            cam_display,
            (WIN_W, WIN_H)
        )

        # ====================================================
        # MAPPING SANG BEV
        # ====================================================

        sat_display = satellite_base.copy()

        mapped_vehicles = (
            bev_mapper.draw_vehicles(
                sat_display,
                vehicles
            )
        )

        # ====================================================
        # HIỂN THỊ TỌA ĐỘ
        # ====================================================

        y_text = 30

        for vehicle in mapped_vehicles:

            text = (
                f"ID:{vehicle['id']} "
                f"X:{vehicle['x_meter']:.2f}m "
                f"Y:{vehicle['y_meter']:.2f}m "
                f"[{vehicle['class']}]"
            )

            cv2.putText(
                sat_display,
                text,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            y_text += 22

        # ====================================================
        # HIỂN THỊ FPS
        # ====================================================

        fps_counter += 1

        elapsed = time.time() - fps_timer

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_text = f"FPS: {fps:.1f}"

            cv2.putText(
                cam_display,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                sat_display,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            fps_counter = 0
            fps_timer = time.time()

        # ====================================================
        # HIỂN THỊ THÔNG TIN
        # ====================================================

        info_text = (
            f"Detections: {len(detections)}"
        )

        cv2.putText(
            cam_display,
            info_text,
            (10, WIN_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        info_text = (
            f"Tracked: {len(mapped_vehicles)}"
        )

        cv2.putText(
            sat_display,
            info_text,
            (10, WIN_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ====================================================
        # HIỂN THỊ
        # ====================================================

        cv2.imshow(
            WIN_LEFT,
            cam_display
        )

        cv2.imshow(
            WIN_RIGHT,
            sat_display
        )

        # ====================================================
        # XỬ LÝ PHÍM
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key in (
            27,
            ord("q"),
            ord("Q")
        ):

            break

        # ----------------------------------------------------
        # Nếu đóng cửa sổ
        # ----------------------------------------------------

        try:

            left_visible = cv2.getWindowProperty(
                WIN_LEFT,
                cv2.WND_PROP_VISIBLE
            )

            right_visible = cv2.getWindowProperty(
                WIN_RIGHT,
                cv2.WND_PROP_VISIBLE
            )

            if (
                left_visible < 1
                or right_visible < 1
            ):

                break

        except cv2.error:

            break

    # ========================================================
    # 7. GIẢI PHÓNG
    # ========================================================

    vs.stop()

    tracker.reset()

    cv2.destroyAllWindows()

    print()
    print(
        "[SYSTEM] Da ket thuc."
    )
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()