import argparse
import os

from fight_detection_pose_lstm.image_transformations.base import (
    ImageTransformationPipeline,
)
from fight_detection_pose_lstm.image_transformations.edges import HighlightEdges
from fight_detection_pose_lstm.train import Training, FIGHT_LABEL, NO_FIGHT_LABEL
from models.open_pose_gupta import OpenPoseArgs, OpenPoseGuptaModel
from configs.dataclasses import (
    AngleCalculatorConfig,
    DirectoryConfig,
    DIRECTORY_CONFIG,
    ANGLE_CONFIG,
    CLASSIF_MODEL,
    KEYPOINT_CONFIG,
)
from utils import check_path, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_lstm", description="Run OpenPose on frames to train a Bi-LSTM model"
    )
    parser.add_argument(
        "config", type=str, help="Path to the config for running training"
    )

    args = parser.parse_args()
    check_path(args.config)
    config_dict = load_config(args.config)
    dir_config = DirectoryConfig(**config_dict[DIRECTORY_CONFIG])
    angle_config = AngleCalculatorConfig(**config_dict[ANGLE_CONFIG])
    keypoint_config = OpenPoseArgs(**config_dict[KEYPOINT_CONFIG])

    check_path(dir_config.directory)

    fight_dir = os.path.join(dir_config.directory, dir_config.fight_subdirectory)
    no_fight_dir = os.path.join(dir_config.directory, dir_config.no_fight_subdirectory)
    check_path(fight_dir)
    check_path(no_fight_dir)

    keypoint_model = OpenPoseGuptaModel(keypoint_config)
    transformations = ImageTransformationPipeline([HighlightEdges()])
    training = Training(
        keypoint_model=keypoint_model,
        fight_pairs_indexes=angle_config.fight_pairs_indexes,
        angle_bins=angle_config.angle_bins,
        transformations=transformations,
    )

    fight_sequence_dirs = [
        os.path.join(fight_dir, subdir) for subdir in os.listdir(fight_dir)
    ]
    no_fight_sequence_dirs = [
        os.path.join(no_fight_dir, subdir) for subdir in os.listdir(no_fight_dir)
    ]

    for seq_dir in fight_sequence_dirs:
        training.process_sequence(seq_dir, FIGHT_LABEL)
    for seq_dir in no_fight_sequence_dirs:
        training.process_sequence(seq_dir, NO_FIGHT_LABEL)
