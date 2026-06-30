import argparse
from pathlib import Path
from models.open_pose_gupta import OpenPoseGuptaModel, OpenPoseArgs
from fight_detection_pose_lstm.image_transformations.utils import (
    show_before_after,
    read_image,
)
from fight_detection_pose_lstm.config import ConfigKeys
from fight_detection_pose_lstm.utils import check_path, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test_open_pose",
        description="Run OpenPose on a frame. Shows before and after",
    )
    parser.add_argument("config", type=str, help="Path to the config")

    args = parser.parse_args()
    check_path(args.config)
    config_dict = load_config(args.config)
    if ConfigKeys.keypoint not in config_dict:
        raise AttributeError(f"Missing {ConfigKeys.keypoint}")
    keypoint_config = OpenPoseArgs(**config_dict[ConfigKeys.keypoint])

    args = parser.parse_args()

    model = OpenPoseGuptaModel(
        OpenPoseArgs(
            model_path=keypoint_config.model_path, proto_path=keypoint_config.proto_path
        )
    )

    images = []
    img_path = config_dict["image_path"]
    check_path(img_path)
    path = Path(img_path)
    if path.is_dir():
        images = list(path.glob("*.[pj][np]g"))
    else:
        images = [str(path)]
    if images:
        for image in images:
            image_np = read_image(image)
            keypoints = model.inference(image_np)
            original, drawn = model.draw_keypoints(image_np, keypoints)
            show_before_after(original, drawn)
