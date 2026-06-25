import argparse
from pathlib import Path
from models.open_pose_gupta import OpenPoseGuptaModel, OpenPoseArgs
from fight_detection_pose_lstm.image_transformations.utils import show_before_after, read_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_or_directory")
    parser.add_argument("model_path")
    parser.add_argument("proto_path")

    args = parser.parse_args()

    model = OpenPoseGuptaModel(OpenPoseArgs(local_path=args.model_path, proto_path=args.proto_path))

    images = []
    path = Path(args.image_or_directory)
    if path.is_dir():
        images = list(path.glob('*.[pj][np]g'))
    else:
        images = [str(path)]
    if images:
        for image in images:
            image_np = read_image(image)
            keypoints = model.inference(image_np)
            original, drawn = model.draw_keypoints(image_np, keypoints)
            show_before_after(original, drawn)

