import cv2
from abc import ABC, abstractmethod


class ImageConverter(ABC):
    @abstractmethod
    def convert(self, image):
        pass


class BayerToBGRConverter(ImageConverter):
    def convert(self, bayer_image):
        # Assuming bayer_image is in the format of numpy array
        return cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2BGR)


if __name__ == '__main__':
    # Test the converter with a sample Bayer image
    sample_bayer_image = cv2.imread('sample_bayer_image.png', cv2.IMREAD_UNCHANGED)
    converter = BayerToBGRConverter()
    bgr_image = converter.convert(sample_bayer_image)
    cv2.imwrite('output_image.png', bgr_image)
