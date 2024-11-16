import cv2
from threading import Thread


class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=15):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        ret = self.stream.set(cv2.CAP_PROP_GAIN, 100)
        ret = self.stream.set(cv2.CAP_PROP_FPS, framerate)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        return self.frame

    def stop(self):
        self.stopped = True
