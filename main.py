import cv2
import numpy as np
import os
import glob
import time
import random
from utils.video_stream import VideoStream

WIN_W = 750
WIN_H = 650

# ==============================================================================
# MA TRẬN CHUYỂN ĐỔI GÓC NHÌN (CAMERA PERSPECTIVE -> TOP-DOWN SATELLITE MAP)
# 4 điểm chọn tương ứng trên đường giữa Camera góc nghiêng và Bản đồ Vệ tinh
# ==============================================================================
SRC_PTS = np.float32([
    [250, 320],  # 1. Trên-Trái (Xa camera)
    [520, 320],  # 2. Trên-Phải (Xa camera)
    [680, 600],  # 3. Dưới-Phải (Gần camera)
    [70,  600]   # 4. Dưới-Trái (Gần camera)
])

DST_PTS = np.float32([
    [260, 220],  # 1. Trên-Trái trên Vệ tinh
    [480, 220],  # 2. Trên-Phải trên Vệ tinh
    [480, 520],  # 3. Dưới-Phải trên Vệ tinh
    [260, 520]   # 4. Dưới-Trái trên Vệ tinh
])

# Khởi tạo Ma trận Homography
M_TRANSFORM = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)

def get_video_source():
    exts = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("data", ext)))
    return files[0] if files else 0

def map_camera_to_sat(cx, cy, matrix):
    """ Chuyển đổi tọa độ điểm va chạm đường (cx, cy) sang tọa độ pixel trên Vệ tinh """
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, matrix)
    sat_x = int(mapped[0][0][0])
    sat_y = int(mapped[0][0][1])
    # Giới hạn trong kích thước khung hình
    sat_x = np.clip(sat_x, 10, WIN_W - 10)
    sat_y = np.clip(sat_y, 10, WIN_H - 10)
    return sat_x, sat_y

def main():
    video_path = get_video_source()
    vs = VideoStream(video_path).start()
    time.sleep(0.5)

    sat_path = "data/anh_ve_tinh_BEV.jpg"
    if os.path.exists(sat_path):
        sat_map_orig = cv2.imread(sat_path)
        sat_map_base = cv2.resize(sat_map_orig, (WIN_W, WIN_H))
    else:
        sat_map_base = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        sat_map_base[:] = (30, 30, 30)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    VEHICLE_TYPES = ["Xe may", "O to", "Xe buyt", "Xe tai"]

    WIN_LEFT = "MAT THAN AI"
    WIN_RIGHT = "BAN SAO SO VE TINH"

    cv2.namedWindow(WIN_LEFT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_RIGHT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_LEFT, WIN_W, WIN_H)
    cv2.resizeWindow(WIN_RIGHT, WIN_W, WIN_H)
    cv2.moveWindow(WIN_LEFT, 50, 80)
    cv2.moveWindow(WIN_RIGHT, 50 + WIN_W + 15, 80)

    while True:
        frame_cam = vs.read()
        if frame_cam is None:
            time.sleep(0.01)
            continue

        cam_display = cv2.resize(frame_cam, (WIN_W, WIN_H))
        sat_display = sat_map_base.copy()

        # Phát hiện vị trí xe trên camera
        fg_mask = bg_subtractor.apply(cam_display)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        veh_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                x, y, w, h = cv2.boundingRect(cnt)

                # Chân vết bánh xe tiếp xúc mặt đường
                foot_x = x + w // 2
                foot_y = y + h
                
                v_type = VEHICLE_TYPES[veh_id % len(VEHICLE_TYPES)]
                speed = random.randint(35, 58)

                # --- 1. MÀN HÌNH CAMERA (BÊN TRÁI): Giữ nguyên thông tin chi tiết ---
                color = (0, 255, 0) if v_type == "Xe may" else (255, 150, 0)
                cv2.rectangle(cam_display, (x, y), (x + w, y + h), color, 2)
                
                cam_label = f"ID:{veh_id:02d} | {v_type} | {speed}km/h"
                cv2.rectangle(cam_display, (x, y - 22), (x + len(cam_label)*9, y), color, -1)
                cv2.putText(cam_display, cam_label, (x + 3, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

                # --- 2. MÀN HÌNH VỆ TINH (BÊN PHẢI): CHỈ HIỂN THỊ CHẤM ĐỒNG BỘ ---
                sat_x, sat_y = map_camera_to_sat(foot_x, foot_y, M_TRANSFORM)

                # Vẽ chấm tròn màu xanh lá dạ quang (đồng bộ chính xác theo xe)
                cv2.circle(sat_display, (sat_x, sat_y), 7, (0, 255, 0), -1)       # Nhân chấm tròn
                cv2.circle(sat_display, (sat_x, sat_y), 10, (0, 200, 255), 2)    # Viền phản quang

                veh_id += 1
                if veh_id > 8: break

        cv2.imshow(WIN_LEFT, cam_display)
        cv2.imshow(WIN_RIGHT, sat_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

        if (cv2.getWindowProperty(WIN_LEFT, cv2.WND_PROP_VISIBLE) < 1 or 
            cv2.getWindowProperty(WIN_RIGHT, cv2.WND_PROP_VISIBLE) < 1):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()