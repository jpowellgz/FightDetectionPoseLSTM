import numpy as np
from src.fightdetectionposelstm.image_transformations.base import ImageTransformation
from src.fightdetectionposelstm.image_transformations.constants import SOBEL_LR_KERNEL, SOBEL_RL_KERNEL



class HighlightEdges(ImageTransformation):
    """Calculates the Sobel edges of an image and darkens them over the original"""
    def __init__(self, weight: float=0.3):
        self.weight = weight

    def transformations(self, image) -> np.ndarray:
        sobel_lr = self.apply_filter(image, SOBEL_LR_KERNEL, -1)
        sobel_rl = self.apply_filter(image, SOBEL_RL_KERNEL, -1)
        added_sobels = self.add_weighted_images(sobel_lr, sobel_rl, 0.5, 0.5)
        highlighted = self.add_weighted_images(image, added_sobels,1 - self.weight,  self.weight)
        return highlighted
