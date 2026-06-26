import argparse
import os

from fight_detection_pose_lstm.image_transformations.base import ImageTransformationPipeline
from fight_detection_pose_lstm.image_transformations.edges import HighlightEdges
from fight_detection_pose_lstm.train import Training, FIGHT_LABEL, NO_FIGHT_LABEL
from models.open_pose_gupta import OpenPoseArgs, OpenPoseGuptaModel, OPENPOSE_FIGHT_PAIRS

def check_dir(directory: str):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"{directory} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_lstm", description="Run OpenPose on frames to train a Bi-LSTM model")
    parser.add_argument("directory", type=str, help="Path to the root directory of the dataset")
    parser.add_argument("fight_subdirectory", type=str, help="Name of the fight subdirectory")
    parser.add_argument("no_fight_subdirectory", type=str, help="Name of the no-fight subdirectory")
    parser.add_argument("model_path", type=str, help="Path to Openpose model")
    parser.add_argument("proto_path", type=str, help="Path to Openpose proto path")
    parser.add_argument("--angle_bins", type=int, default=20, help="Number of angle bins")

    args = parser.parse_args()
    check_dir(args.directory)

    fight_dir = os.path.join(args.directory, args.fight_subdirectory)
    no_fight_dir = os.path.join(args.directory, args.no_fight_subdirectory)
    check_dir(fight_dir)
    check_dir(no_fight_dir)


    keypoint_model = OpenPoseGuptaModel(OpenPoseArgs(local_path=args.model_path, proto_path=args.proto_path))
    transformations = ImageTransformationPipeline([HighlightEdges()])
    training = Training(
        keypoint_model = keypoint_model,
        fight_pairs_indexes = OPENPOSE_FIGHT_PAIRS,
        angle_bins = args.angle_bins,
        transformations=transformations,
    )

    fight_sequence_dirs = [os.path.join(fight_dir, subdir) for subdir in os.listdir(fight_dir)]
    no_fight_sequence_dirs = [os.path.join(no_fight_dir, subdir) for subdir in os.listdir(no_fight_dir)]

    for seq_dir in fight_sequence_dirs:
        training.process_sequence(seq_dir, FIGHT_LABEL)
    for seq_dir in no_fight_sequence_dirs:
        training.process_sequence(seq_dir, NO_FIGHT_LABEL)
    
    



            
            



