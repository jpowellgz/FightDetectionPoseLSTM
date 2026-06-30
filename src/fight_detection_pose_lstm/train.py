from dataclasses import dataclass
import os

import numpy as np

from sklearn.model_selection import train_test_split
from fight_detection_pose_lstm.image_transformations.utils import read_image
from fight_detection_pose_lstm.image_transformations.base import (
    ImageTransformationPipeline,
)
from fight_detection_pose_lstm.model_base import KeypointModel
from fight_detection_pose_lstm.skeletons import AngleCalculator, Skeleton
from fight_detection_pose_lstm.config import AngleCalculatorConfig


@dataclass
class Labels:
    fight: int = 1
    no_fight: int = 0


@dataclass
class Sequence:
    images: list[np.ndarray]
    skeletons: list[Skeleton]
    label: int
    vectors: list[np.ndarray]


class Training:
    def __init__(
        self,
        keypoint_model: KeypointModel,
        angle_calculator_config: AngleCalculatorConfig,
        transformations: ImageTransformationPipeline | None = None,
    ):
        self.keypoint_model = keypoint_model
        self.transformations = transformations
        self.angle_config = angle_calculator_config
        self.sequences: list[Sequence] = []

    def process_sequence(self, directory: str, label: int):
        sequence = self.get_sequence_values(directory, label)
        self.sequences.append(sequence)

    def get_sequence_values(self, frame_dir: str, label: int):
        frames = [
            os.path.join(frame_dir, frame_name) for frame_name in os.listdir(frame_dir)
        ]
        seq = Sequence([], [], label, [])
        for frame in frames:
            angle_calculator = AngleCalculator(
                self.angle_config.angle_bins, len(self.angle_config.fight_pairs_indexes)
            )
            image_np = read_image(frame)
            if self.transformations is not None:
                image_np = self.transformations.transform_image(image_np)
            skeletons = self.keypoint_model.infer_skeletons(
                image_np, fight_pairs_indexes=self.angle_config.fight_pairs_indexes
            )
            for skeleton in skeletons:
                angle_calculator.add_skeleton_distribution(skeleton)
            vector = angle_calculator.get_distribution_vector()
            seq.images.append(image_np)
            seq.skeletons.append(skeletons)
            seq.vectors.append(vector)
        return seq

    def sequences_to_training_data(self):
        data = []
        labels = []
        for sequence in self.sequences:
            vectors = np.asarray(sequence.vectors)
            data.append(vectors)
            labels.append(sequence.label)
        data_np = np.asarray(data)
        labels_np = np.asarray(labels)

        x_train,x_test,y_train,y_test=train_test_split(data_np,labels_np,train_size=self.args.train_size)
        return x_train, x_test, y_train, y_test