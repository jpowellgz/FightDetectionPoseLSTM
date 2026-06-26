from dataclasses import dataclass
import os

import numpy as np
from fight_detection_pose_lstm.image_transformations.utils import read_image
from fight_detection_pose_lstm.image_transformations.base import ImageTransformationPipeline
from fight_detection_pose_lstm.model_base import KeypointModel
from fight_detection_pose_lstm.skeletons import AngleCalculator, Skeleton
from fight_detection_pose_lstm.logging import logger

FIGHT_LABEL = 1
NO_FIGHT_LABEL = 0


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
        fight_pairs_indexes: list[int],
        angle_bins: int,
        transformations: ImageTransformationPipeline | None = None,
    ):
        self.keypoint_model = keypoint_model
        self.transformations = transformations
        self.fight_pairs_indexes = fight_pairs_indexes
        self.angle_bins = angle_bins
        self.sequences = []

    def process_sequence(self, directory: str, label: int):
        sequence = self.get_sequence_values(directory, label)
        self.sequences.append(sequence)
        logger.info(self.sequences[0])

    def get_sequence_values(self, frame_dir: str, label: int):
        frames = [
            os.path.join(frame_dir, frame_name) for frame_name in os.listdir(frame_dir)
        ]
        seq = Sequence([], [], label, [])
        for frame in frames:
            angle_calculator = AngleCalculator(
                self.angle_bins, len(self.fight_pairs_indexes)
            )
            image_np = read_image(frame)
            if self.transformations is not None:
                image_np = self.transformations.transform_image(image_np)
            skeletons = self.keypoint_model.infer_skeletons(
                image_np, fight_pairs_indexes=self.fight_pairs_indexes
            )
            for skeleton in skeletons:
                angle_calculator.add_skeleton_distribution(skeleton)
            vector = angle_calculator.get_distribution_vector()
            seq.images.append(image_np)
            seq.skeletons.append(skeletons)
            seq.vectors.append(vector)
        return seq
