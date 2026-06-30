import argparse
import os
from typing import Any

from fight_detection_pose_lstm.image_transformations.base import (
    ImageTransformationPipeline,
)
from fight_detection_pose_lstm.image_transformations.edges import HighlightEdges
from fight_detection_pose_lstm.train import Training, Labels
from models.open_pose_gupta import OpenPoseArgs, OpenPoseGuptaModel
from models.tensorflow_lstm import LSTMArgs, TensorflowLSTM
from fight_detection_pose_lstm.config import (
    AngleCalculatorConfig,
    DirectoryConfig,
    ConfigKeys,
)
from fight_detection_pose_lstm.utils import check_path, load_config


def check_config(config: dict[str, Any]) -> None:
    if ConfigKeys.directory not in config:
        raise AttributeError(f"Missing {ConfigKeys.directory}")
    if ConfigKeys.angle not in config:
        raise AttributeError(f"Missing {ConfigKeys.angle}")
    if ConfigKeys.keypoint not in config:
        raise AttributeError(f"Missing {ConfigKeys.keypoint}")
    if ConfigKeys.classification not in config:
        raise AttributeError(f"Missing {ConfigKeys.classification}")

def init_models(config: dict[str, Any]):
    keypoint_config = OpenPoseArgs(**config_dict[ConfigKeys.keypoint])
    classification_config = LSTMArgs(**config_dict[ConfigKeys.classification])
    kpt = OpenPoseGuptaModel(keypoint_config)
    classif = TensorflowLSTM(classification_config)

    return kpt, classif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_lstm", description="Run OpenPose on frames to train a Bi-LSTM model"
    )
    parser.add_argument(
        "config", type=str, help="Path to the config for running training"
    )

    # Check directory
    args = parser.parse_args()
    check_path(args.config)
    config_dict = load_config(args.config)
    check_config(config_dict)
    dir_config = DirectoryConfig(**config_dict[ConfigKeys.directory])
    angle_config = AngleCalculatorConfig(**config_dict[ConfigKeys.angle])


    check_path(dir_config.directory)

    fight_dir = os.path.join(dir_config.directory, dir_config.fight_subdirectory)
    no_fight_dir = os.path.join(dir_config.directory, dir_config.no_fight_subdirectory)
    check_path(fight_dir)
    check_path(no_fight_dir)

    keypoint_model, classification_model = init_models(config_dict)
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
        training.process_sequence(seq_dir, Labels.fight)
    for seq_dir in no_fight_sequence_dirs:
        training.process_sequence(seq_dir, Labels.no_fight)
