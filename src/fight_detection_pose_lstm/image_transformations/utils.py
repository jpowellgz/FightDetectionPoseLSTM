import os
import cv2
import numpy as np

from fight_detection_pose_lstm.image_transformations.base import ImageTransformation, ImageTransformationPipeline

def show_before_after(image_before:np.ndarray, image_after: np.ndarray, scale: float=1.0, horizontal=True):
    """Method to show two images side by side or one on top of the other

    Args:
        image_before (_type_): image before
        image_after (_type_): _description_
        scale (int, optional): _description_. Defaults to 1.
        horizontal (bool, optional): _description_. Defaults to True.
    """
    axis = 1 if horizontal else 0
    image = np.concatenate((image_before, image_after), axis=axis)
    shape = [int(dim * scale) for dim in image.shape] if scale != 1.0 else image.shape
    new = cv2.resize(image, (shape[1], shape[0]))
    cv2.imshow("Before / After", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_image(image_path: str) -> np.ndarray | None:
    if os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        print(f"Image {image_path} doesn't exist")
        return None

def test_with_image(image_path: str, transformation: ImageTransformation | ImageTransformationPipeline, scale:float=1.0):
    """Test a function with an image

    Args:
        image_path (str): path to image
        function (callable): function to test
    """
    cv2_image = read_image(image_path)
    if cv2_image is not None:
        output = transformation.transform_image(cv2_image)
        show_before_after(cv2_image, output, scale)

def show_image(image:np.ndarray):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
