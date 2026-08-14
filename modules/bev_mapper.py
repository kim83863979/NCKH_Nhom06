import cv2
import numpy as np


class BEVMapper:

    def __init__(
        self,
        homography_matrix,
        bev_width,
        bev_height,
        scale=20
    ):

        self.H = homography_matrix

        self.bev_width = bev_width
        self.bev_height = bev_height

        self.scale = scale

    # ==========================================================
    # PIXEL CAMERA -> TỌA ĐỘ BEV
    # ==========================================================
    def pixel_to_bev(self, u, v):

        point = np.array(
            [[[u, v]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            point,
            self.H
        )

        x = float(
            mapped[0][0][0]
        )

        y = float(
            mapped[0][0][1]
        )

        return x, y

    # ==========================================================
    # NHIỀU ĐIỂM
    # ==========================================================
    def pixels_to_bev(self, points):

        if not points:
            return []

        pts = np.array(
            points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        mapped = cv2.perspectiveTransform(
            pts,
            self.H
        )

        return mapped.reshape(-1, 2)

    # ==========================================================
    # BEV PIXEL -> MET
    # ==========================================================
    def bev_to_meter(self, x, y):

        real_x = x / self.scale
        real_y = y / self.scale

        return real_x, real_y

    # ==========================================================
    # PIXEL CAMERA -> MÉT
    # ==========================================================
    def pixel_to_meter(self, u, v):

        x, y = self.pixel_to_bev(
            u,
            v
        )

        real_x, real_y = self.bev_to_meter(
            x,
            y
        )

        return real_x, real_y

    # ==========================================================
    # GIỚI HẠN TỌA ĐỘ
    # ==========================================================
    def clamp_position(self, x, y):

        x = max(
            0,
            min(
                int(x),
                self.bev_width - 1
            )
        )

        y = max(
            0,
            min(
                int(y),
                self.bev_height - 1
            )
        )

        return x, y

    # ==========================================================
    # VẼ XE MÁY
    # ==========================================================
    def draw_motorbike(
        self,
        image,
        x,
        y,
        color=(0, 255, 0),
        radius=7
    ):

        x, y = self.clamp_position(
            x,
            y
        )

        cv2.circle(
            image,
            (x, y),
            radius,
            color,
            -1
        )

        cv2.circle(
            image,
            (x, y),
            radius + 3,
            color,
            2
        )

    # ==========================================================
    # VẼ Ô TÔ
    # ==========================================================
    def draw_car(
        self,
        image,
        x,
        y,
        color=(255, 0, 0),
        width=30,
        height=50
    ):

        x, y = self.clamp_position(
            x,
            y
        )

        x1 = int(
            x - width / 2
        )

        y1 = int(
            y - height / 2
        )

        x2 = int(
            x + width / 2
        )

        y2 = int(
            y + height / 2
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            -1
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

    # ==========================================================
    # VẼ PHƯƠNG TIỆN
    # ==========================================================
    def draw_vehicle(
        self,
        image,
        vehicle
    ):

        vehicle_id = vehicle.get(
            "id",
            -1
        )

        vehicle_class = vehicle.get(
            "class",
            ""
        )

        u = vehicle.get(
            "u",
            0
        )

        v = vehicle.get(
            "v",
            0
        )

        # ------------------------------------------------------
        # CAMERA PIXEL -> BEV
        # ------------------------------------------------------
        x, y = self.pixel_to_bev(
            u,
            v
        )

        # ------------------------------------------------------
        # VẼ THEO CLASS
        # ------------------------------------------------------
        class_lower = str(
            vehicle_class
        ).lower()

        if (
            "motor" in class_lower
            or "bike" in class_lower
            or "xe may" in class_lower
            or "xe máy" in class_lower
        ):

            self.draw_motorbike(
                image,
                x,
                y
            )

        else:

            self.draw_car(
                image,
                x,
                y
            )

        # ------------------------------------------------------
        # HIỂN THỊ ID
        # ------------------------------------------------------
        draw_x, draw_y = self.clamp_position(
            x,
            y
        )

        cv2.putText(
            image,
            f"ID:{vehicle_id}",
            (
                draw_x + 10,
                draw_y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # ------------------------------------------------------
        # TRẢ VỀ TỌA ĐỘ
        # ------------------------------------------------------
        return {
            "id": vehicle_id,
            "class": vehicle_class,
            "x": float(x),
            "y": float(y),
            "x_meter": float(
                x / self.scale
            ),
            "y_meter": float(
                y / self.scale
            )
        }

    # ==========================================================
    # VẼ TOÀN BỘ XE
    # ==========================================================
    def draw_vehicles(
        self,
        image,
        vehicles
    ):

        results = []

        for vehicle in vehicles:

            result = self.draw_vehicle(
                image,
                vehicle
            )

            results.append(
                result
            )

        return results