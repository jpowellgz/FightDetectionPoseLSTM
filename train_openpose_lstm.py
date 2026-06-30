import argparse
import os
from typing import Any

import numpy as np
import tqdm
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
    TrainingConfig,
)
from fight_detection_pose_lstm.logging import logger
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
    if ConfigKeys.training not in config:
        raise AttributeError(f"Missing {ConfigKeys.training}")

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
    train_config = TrainingConfig(**config_dict[ConfigKeys.training])

    # Check paths
    check_path(dir_config.directory)

    fight_dir = os.path.join(dir_config.directory, dir_config.fight_subdirectory)
    no_fight_dir = os.path.join(dir_config.directory, dir_config.no_fight_subdirectory)
    check_path(fight_dir)
    check_path(no_fight_dir)

    # Start models
    keypoint_model, classification_model = init_models(config_dict)
    transformations = ImageTransformationPipeline([HighlightEdges()])
    training = Training(
        keypoint_model=keypoint_model,
        angle_calculator_config=angle_config,
        transformations=transformations,
        train_size=train_config.train_size,
    )

    if train_config.load_training_data:
        with np.load(train_config.load_path) as data:
            x_train = data["x_train"]
            x_test = data["x_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
    else:

        # Process each sequence of frames
        fight_sequence_dirs = [
            os.path.join(fight_dir, subdir) for subdir in os.listdir(fight_dir)
        ]
        no_fight_sequence_dirs = [
            os.path.join(no_fight_dir, subdir) for subdir in os.listdir(no_fight_dir)
        ]

        logger.info("Processing fight sequences")
        for seq_dir in tqdm.tqdm(fight_sequence_dirs):
            training.process_sequence(seq_dir, Labels.fight)
        logger.info("Processing nonfight sequences")
        for seq_dir in tqdm.tqdm(no_fight_sequence_dirs):
            training.process_sequence(seq_dir, Labels.no_fight)
    
        x_train, x_test, y_train, y_test = training.sequences_to_training_data()
        if train_config.save_training_data:
            np.savez(train_config.save_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    classification_model.train(x_train, y_train)
    score = classification_model.evaluate(x_test, y_test)
    logger.info(f"Classification Score: {score[1]}")

