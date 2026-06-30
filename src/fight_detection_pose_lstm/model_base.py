from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np

from fight_detection_pose_lstm.skeletons import Skeleton


@dataclass
class ModelArgs:
    model_path: str | None


class Model(ABC):
    """Base class to get a model, train it on vectors and perform inference"""

    def __init__(self, model_args: ModelArgs):
        self.args = model_args

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def inference(self, x_input: np.ndarray) -> np.ndarray:
        pass


@dataclass
class KeypointModelArgs(ModelArgs):
    num_keypoints: int
    keypoint_names: list[str]
    pairs: list[list[int]]


class KeypointModel(Model):
    def __init__(self, model_args: KeypointModelArgs):
        super().__init__(model_args)

    def infer_skeletons(self, x_input: np.ndarray, fight_pairs_indexes: list[int]) -> list[Skeleton]:
        keypoints = self.inference(x_input)
        skeletons = []
        for n in range(keypoints.shape[0]):
            skeleton = Skeleton(
                num_keypoints=self.args.num_keypoints,
                keypoints_np=keypoints[n],
                keypoint_names=self.args.keypoint_names,
                pairs=self.args.pairs,
                fight_pairs_indexes=fight_pairs_indexes,
            )
            skeletons.append(skeleton)
        return skeletons
