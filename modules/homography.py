import cv2
import numpy as np
import os
import sys


class Homography:
    def __init__(self):
        self.src_points = []
        self.dst_points = None
        self.matrix = None

    # ==========================================================
    # CALLBACK CLICK CHUỘT
    # ==========================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            # Chỉ lấy tối đa 4 điểm
            if len(self.src_points) >= 4:
                return

            self.src_points.append([x, y])

            print(
                f"[HOMOGRAPHY] Điểm {len(self.src_points)}: "
                f"({x}, {y})"
            )

            # Vẽ điểm lên ảnh
            cv2.circle(
                self.image_display,
                (x, y),
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                self.image_display,
                str(len(self.src_points)),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(self.window_name, self.image_display)

    # ==========================================================
    # CHỌN 4 ĐIỂM TRÊN ẢNH
    # ==========================================================
    def select_points(self, image):
        self.src_points = []

        self.window_name = "CHON 4 DIEM HOMOGRAPHY"

        self.image_display = image.copy()

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )

        print("\n======================================")
        print(" CHON 4 DIEM HOMOGRAPHY")
        print("======================================")
        print("Click 4 diem theo thu tu:")
        print("1. Tren - Trai")
        print("2. Tren - Phai")
        print("3. Duoi - Phai")
        print("4. Duoi - Trai")
        print()
        print("Nhan ENTER de xac nhan.")
        print("Nhan ESC de huy.")
        print("======================================")

        while True:

            cv2.imshow(
                self.window_name,
                self.image_display
            )

            key = cv2.waitKey(1) & 0xFF

            # ENTER
            if key == 13:

                if len(self.src_points) == 4:
                    break

                print(
                    f"[WARNING] Hien tai co "
                    f"{len(self.src_points)}/4 diem."
                )

            # ESC
            elif key == 27:
                cv2.destroyWindow(self.window_name)
                return None

        cv2.destroyWindow(self.window_name)

        return np.float32(self.src_points)

    # ==========================================================
    # TẠO HỆ TỌA ĐỘ ĐÍCH
    # ==========================================================
    def create_destination_points(
        self,
        width_meters,
        height_meters,
        scale=100
    ):
        """
        width_meters  : chiều rộng thực tế của vùng đường
        height_meters : chiều dài thực tế của vùng đường
        scale         : số pixel biểu diễn 1 mét

        Ví dụ:

        width = 20m
        height = 30m
        scale = 20

        => ảnh BEV = 400 x 600 pixel
        """

        width_px = int(width_meters * scale)
        height_px = int(height_meters * scale)

        self.dst_points = np.float32([
            [0, 0],
            [width_px, 0],
            [width_px, height_px],
            [0, height_px]
        ])

        return self.dst_points

    # ==========================================================
    # TÍNH MA TRẬN HOMOGRAPHY
    # ==========================================================
    def calculate(self):

        if len(self.src_points) != 4:
            raise ValueError(
                "Phai co dung 4 diem nguon."
            )

        if self.dst_points is None:
            raise ValueError(
                "Chua tao diem dich."
            )

        self.matrix = cv2.getPerspectiveTransform(
            np.float32(self.src_points),
            np.float32(self.dst_points)
        )

        print("\n======================================")
        print(" MA TRAN HOMOGRAPHY")
        print("======================================")
        print(self.matrix)
        print("======================================")

        return self.matrix

    # ==========================================================
    # CHUYỂN 1 ĐIỂM PIXEL -> BEV
    # ==========================================================
    def transform_point(self, u, v):

        if self.matrix is None:
            raise ValueError(
                "Chua tinh ma tran Homography."
            )

        point = np.array(
            [[[u, v]]],
            dtype=np.float32
        )

        result = cv2.perspectiveTransform(
            point,
            self.matrix
        )

        x = float(result[0][0][0])
        y = float(result[0][0][1])

        return x, y

    # ==========================================================
    # CHUYỂN NHIỀU ĐIỂM
    # ==========================================================
    def transform_points(self, points):

        if self.matrix is None:
            raise ValueError(
                "Chua tinh ma tran Homography."
            )

        if len(points) == 0:
            return []

        pts = np.array(
            points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        result = cv2.perspectiveTransform(
            pts,
            self.matrix
        )

        result = result.reshape(-1, 2)

        return result

    # ==========================================================
    # LƯU MA TRẬN
    # ==========================================================
    def save(self, path):

        if self.matrix is None:
            raise ValueError(
                "Chua co ma tran de luu."
            )

        folder = os.path.dirname(path)

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        np.save(path, self.matrix)

        print(
            f"[HOMOGRAPHY] Da luu: {path}"
        )

    # ==========================================================
    # ĐỌC MA TRẬN
    # ==========================================================
    def load(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )

        self.matrix = np.load(path)

        print(
            f"[HOMOGRAPHY] Da load: {path}"
        )

        return self.matrix


# ==============================================================
# CHẠY TEST ĐỘC LẬP
# ==============================================================
if __name__ == "__main__":

    image_path = "data/anh_ngatu_tinh.jpg"

    if not os.path.exists(image_path):
        print(
            "Khong tim thay anh:",
            image_path
        )
        exit()

    image = cv2.imread(image_path)

    if image is None:
        print("Khong doc duoc anh.")
        exit()

    homography = Homography()

    # ----------------------------------------------------------
    # BƯỚC 1: CLICK 4 ĐIỂM
    # ----------------------------------------------------------
    points = homography.select_points(image)

    if points is None:
        print("Da huy.")
        exit()

    print("\n4 diem da chon:")
    print(points)

    # ----------------------------------------------------------
    # BƯỚC 2: NHẬP KÍCH THƯỚC THỰC TẾ
    # ----------------------------------------------------------
    try:

        width = float(
            input(
                "\nNhap chieu rong thuc te (met): "
            )
        )

        height = float(
            input(
                "Nhap chieu dai thuc te (met): "
            )
        )

    except ValueError:

        print(
            "Kich thuoc khong hop le."
        )

        exit()

    # ----------------------------------------------------------
    # BƯỚC 3: TẠO ĐIỂM ĐÍCH
    # ----------------------------------------------------------
    homography.create_destination_points(
        width,
        height,
        scale=20
    )

    # ----------------------------------------------------------
    # BƯỚC 4: TÍNH MA TRẬN
    # ----------------------------------------------------------
    matrix = homography.calculate()

    # ----------------------------------------------------------
    # BƯỚC 5: LƯU
    # ----------------------------------------------------------
    homography.save(
        "data/homography.npy"
    )

    print(
        "\nHoan thanh."
    )