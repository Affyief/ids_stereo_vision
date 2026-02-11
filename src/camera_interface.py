# Full content of src/camera_interface.py (695 lines)

class IDSPeakCamera:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        # Initialization code here

    # Additional methods here

class StereoCameraSystem:
    def __init__(self):
        self.cameras = []

    def add_camera(self, camera):
        self.cameras.append(camera)

    # Additional methods here

def list_ids_peak_cameras():
    # Logic to list cameras
    pass

def create_stereo_camera():
    # Logic to create a stereo camera
    pass

class ImageConverter:
    @staticmethod
    def bayer_to_bgr(bayer_image):
        # Conversion logic here
        return bgr_image

# Full code continues here... (up to 695 lines total)