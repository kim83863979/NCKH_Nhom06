import cv2
from threading import Thread

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Chạy luồng đọc video ngầm để không gây giật lag UI
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            # Tự động lặp lại video khi hết
            if not self.grabbed:
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()