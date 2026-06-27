from dataclasses import dataclass
import math
import numpy as np
from fight_detection_pose_lstm.logging import logger


@dataclass
class Keypoint:
    """Base Keypoint class"""
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def is_valid(self) -> bool:
        if self.x > 0 and self.y > 0:
            return True
        return False


class Skeleton:
    """Base Skeleton class"""
    keypoints_np: np.ndarray
    num_keypoints: int
    keypoint_names: list[str]
    pairs: list[list[int]]
    fight_pairs_indexes: list[int]

    def __init__(
        self,
        num_keypoints: int,
        keypoints_np: np.ndarray,
        keypoint_names: list[str] | None = None,
        pairs: list[list[int]] | None = None,
        fight_pairs_indexes: list[int] | None = None
    ):
        self.num_keypoints = num_keypoints
        self.keypoints = [Keypoint(0, 0) for _ in range(self.num_keypoints)]
        for idx, kpt in enumerate(keypoints_np):
            self.set_keypoint(idx, int(kpt[0]), int(kpt[1]))
        self.keypoint_names = keypoint_names if keypoint_names is not None else []
        
        self.pairs = pairs if pairs is not None else []
        self.fight_pairs_indexes = fight_pairs_indexes if fight_pairs_indexes is not None else []


    def set_keypoint(self, kpt_index: int, x: int, y: int) -> None:
        """Set a keypoint

        Args:
            kpt_index (int): index of the keypoint
            x (int): x value
            y (int): y value
        """
        self.keypoints[kpt_index] = Keypoint(x, y)

    def get_keypoint_pair(self, pair_idx: int) -> tuple[Keypoint]:
        """Get the values of a pair of keypoints from the index number of the pair

        Args:
            pair_idx (int): index of the pair

        Returns:
            tuple[Keypoint]: pair of keypoints
        """
        kpt_one_index = self.pairs[pair_idx][0]
        kpt_two_index = self.pairs[pair_idx][1]
        keypoint_one = self.keypoints[kpt_one_index]
        keypoint_two = self.keypoints[kpt_two_index]
        return keypoint_one, keypoint_two




class AngleCalculator:
    def __init__(self, angle_bins: int = 20, num_limbs: int= 13):
        self.angles = [k / angle_bins for k in range(angle_bins)]
        self.angle_sin = [math.sin(angle*2*math.pi) for angle in self.angles]
        self.angle_cos = [math.cos(angle*2*math.pi) for angle in self.angles]
        self.angle_distribution = np.zeros((num_limbs, angle_bins))

    @staticmethod
    def get_angle_index(sin_indexes: list[int], cos_indexes: list[int]) -> int:
        """Get the best match of angle index from the cosine and sine indexes

        Args:
            sin_indexes (list[int]): list of matching sine indexes
            cos_indexes (list[int]): list of matching cosine indexes

        Returns:
            int: best angle match index
        """
        if len(sin_indexes) == 1 and len(cos_indexes) == 1:
            return sin_indexes[0]
        else:
            s_idx = -1
            for i in range(len(sin_indexes)):
                for c_idx in cos_indexes:
                    if sin_indexes[i] == c_idx:
                        s_idx = sin_indexes[i]
                        break
            if s_idx == -1:
                min_dist = 1000
                for i in range(len(sin_indexes)):
                    for c_idx in cos_indexes:
                        dist = abs(sin_indexes[i] - c_idx)
                        if dist < min_dist:
                            min_dist = dist
                            s_idx = sin_indexes[i]
            return s_idx

    @staticmethod
    def get_indexes(list_values: list[float], trig_value: float) -> list[int]:
        """Get the indexes of possible angles for the trigonometric value given

        Args:
            list_values (list[float]): list of fixed trigonometric values
            trig_value (float): given trigonometric value

        Returns:
            list[int]: list of indexes that match the value
        """
        indexes = []
        length = len(list_values)
        for i in range(length):
            values = [list_values[i], list_values[(i+1)%length]]
            if values[0] < values[1]: 
                if trig_value >= values[0] and trig_value < values[1]:
                    indexes.append(i)
            else:
                if trig_value <= values[0] and trig_value > values[1]:
                    indexes.append(i)
        return indexes

    def search_angle(self, sin_val: float, cos_val: float) -> float:
        """Search the closest angle based on the sin and cos values 

        Args:
            sin_val (float): sine value
            cos_val (float): cosine value

        Returns:
            float: approximate angle
        """
        sin_idxs = self.get_indexes(self.angle_sin, sin_val)
        cos_idxs = self.get_indexes(self.angle_cos, cos_val)
        angle_index = self.get_angle_index(sin_idxs, cos_idxs)
        return angle_index

    def calculate_limb_angle_idx(self, keypoint_one: Keypoint, keypoint_two: Keypoint) -> int:
        """Approximate the angle of a limb without trigonometric functions, using
        distribution bins. Retu

        Args:
            pair_idx (int): index number of the pair

        Returns:
            float: angle of limb, where 0 = 0° = 0 radians, 1.0 = 360° = 2pi radians
        """
        x_diff = keypoint_two.x - keypoint_one.x
        y_diff = keypoint_two.y - keypoint_one.y
        hypotenuse = math.sqrt(x_diff**2 + y_diff**2)
        sin = y_diff / hypotenuse
        cos = x_diff / hypotenuse

        angle_index = self.search_angle(sin, cos)
        return angle_index

    def add_skeleton_distribution(self, skeleton: Skeleton):
        """Calculate the angles for one skeleton and add the values to the distribution

        Args:
            skeleton (Skeleton): input skeleton
        """
        for idx, pair_idx in enumerate(skeleton.fight_pairs_indexes):
            keypoint_one, keypoint_two = skeleton.get_keypoint_pair(pair_idx)
            if keypoint_one.is_valid() and keypoint_two.is_valid():
                angle_index = self.calculate_limb_angle_idx(keypoint_one, keypoint_two)
                self.angle_distribution[idx][angle_index] += 1
    
    def get_distribution_vector(self) -> np.ndarray:
        """Normalize and flatten the distribution

        Returns:
            np.ndarray: distribution vector
        """
        normalized_distribution = self.angle_distribution.copy()
        for i in range(self.angle_distribution.shape[0]):
            sum_val = np.sum(self.angle_distribution[i])
            normalized_distribution[i]  = self.angle_distribution[i] / sum_val if sum_val > 0 else self.angle_distribution[i]
        flat_dist = normalized_distribution.flatten()
        return flat_dist