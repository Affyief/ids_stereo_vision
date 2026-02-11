# Example full contents of camera_interface.py from commit d2122c72fe40a42e2a4b82216ba710a4ab3b914f

# Importing necessary libraries
import cv2
import numpy as np

class CameraInterface:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(camera_id)

    def read_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Conversion from Bayer to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            return frame
        else:
            return None

    def release(self):
        self.camera.release()