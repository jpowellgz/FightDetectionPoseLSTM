import cv2
import numpy as np
from src.fightdetectionposelstm.image_transformations.base import ImageTransformation


class FastNonLocalMeans(ImageTransformation):
    def __init__(
        self, h: int, h_color: int, template_window_size: int, search_window_size: int
    ):
        self.h = h
        self.h_color = h_color
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def transformations(self, image: np.ndarray) -> np.ndarray:
        image_8 = image.astype(np.uint8)
        image_8 = cv2.fastNlMeansDenoisingColored(
            image_8,
            None,
            self.h,
            self.h_color,
            self.template_window_size,
            self.search_window_size,
        )
        return image_8.astype(np.float32)
