import argparse
from pathlib import Path
from fight_detection_pose_lstm.image_transformations.base import ImageTransformationPipeline
from fight_detection_pose_lstm.image_transformations.edges import HighlightEdges
from fight_detection_pose_lstm.image_transformations.utils import test_with_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_or_directory")

    args = parser.parse_args()
    darken_edges = HighlightEdges()
    pipeline = ImageTransformationPipeline([darken_edges])
    images = []
    path = Path(args.image_or_directory)
    if path.is_dir():
        images = list(path.glob('*.[pj][np]g'))
    else:
        images = [str(path)]
    if images:
        for image in images:
            test_with_image(image_path=image, transformation=pipeline, scale=1.0)
