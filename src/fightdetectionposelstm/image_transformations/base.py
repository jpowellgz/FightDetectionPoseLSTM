from abc import abstractmethod
import cv2
import numpy as np


class ImageTransformation:
    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """Base method for image transformation.

        Returns:
            _type_: _description_
        """
        self.base_image = image
        self.float_base_image = self.image_to_float(
            self.base_image
        )  # Image for correction operations
        transformed_float_image = self.transformations(self.float_base_image)
        self.transformed_image = self.float_to_image(transformed_float_image)
        return self.transformed_image

    @abstractmethod
    def transformations(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def image_to_float(image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)

    @staticmethod
    def float_to_image(float_image: np.ndarray) -> np.ndarray:
        return np.clip(float_image, a_min=0, a_max=255).astype(np.uint8)

    def linear_transformation(
        self, image: np.ndarray, a: float, b: float
    ) -> np.ndarray:
        float_image = self.image_to_float(image)
        transformed = a * float_image + b
        return self.float_to_image(transformed)

    @staticmethod
    def normalize_image(image: np.ndarray, alpha, beta) -> np.ndarray:
        """Call the OpenCV normalize function

        Args:
            image (np.ndarray): input image

        Returns:
            np.ndarray: normalized image
        """
        return cv2.normalize(
            image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX
        )

    @staticmethod
    def apply_filter(image: np.ndarray, filter: np.ndarray, dim: int):
        return cv2.filter2D(image, dim, filter)

    @staticmethod
    def add_weighted_images(
        image_one: np.ndarray,
        image_two: np.ndarray,
        weight_one: float,
        weight_two: float,
        gamma: float = 0,
    ):
        return cv2.addWeighted(image_one, weight_one, image_two, weight_two, gamma)


class ImageTransformationPipeline(ImageTransformation):
    def __init__(self, transformations: list[ImageTransformation]):
        self.transformations = transformations

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """Base method for image transformation.

        Returns:
            _type_: _description_
        """
        self.base_image = image
        float_image = self.image_to_float(
            self.base_image
        )  # Image for correction operations
        for transformation in self.transformations:
            float_image = transformation.transform_image(float_image)
        self.transformed_image = self.float_to_image(float_image)
        return self.transformed_image
