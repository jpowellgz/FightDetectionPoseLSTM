from dataclasses import dataclass
import math


@dataclass
class Keypoint:
    """Base Keypoint class"""
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


@dataclass
class Skeleton:
    """Base Skeleton class"""
    keypoints: list[Keypoint]
    num_keypoints: int
    keypoint_names: list[str]
    pairs: list[list[int]]

    def __init__(self, num_keypoints: int, angle_bins:int=20):
        self.num_keypoints = num_keypoints
        self.keypoints = [Keypoint(0, 0) for _ in range(num_keypoints)]
        self.keypoint_names = [""]  # Placeholder for different skeleton structures
        self.pairs = []
        self.calculate_angle_tables(angle_bins)

    def calculate_angle_tables(self, angle_bins: int):
        self.angles = [k / angle_bins for k in range(angle_bins)]
        self.angle_sin = [math.sin(angle) for angle in self.angles]
        self.angle_cos = [math.cos(angle) for angle in self.angles]

    def set_keypoint(self, kpt_index: int, x: int, y: int):
        """Set a keypoint

        Args:
            kpt_index (int): index of the keypoint
            x (int): x value
            y (int): y value
        """
        self.keypoints[kpt_index] = Keypoint(x, y)

    def search_angle(self, sin: float, cos: float):
        return 0.0 #Placeholder

    def calculate_limb_angle(self, pair_idx: int) -> float:
        """Approximate the angle of a limb without trigonometric functions, using
        distribution bins. Retu

        Args:
            pair_idx (int): index number of the pair

        Returns:
            float: angle of limb, where 0 = 0° = 0 radians, 1.0 = 360° = 2pi radians
        """
        kpt_one_index = self.pairs[pair_idx][0]
        kpt_two_index = self.pairs[pair_idx][1]
        keypoint_one = self.keypoints[kpt_one_index]
        keypoint_two = self.keypoints[kpt_two_index]
        x_diff = keypoint_two.x - keypoint_one.x
        y_diff = keypoint_two.y - keypoint_one.y
        hypotenuse = math.sqrt(x_diff**2 + y_diff**2)
        sin = y_diff / hypotenuse
        cos = x_diff / hypotenuse
        angle = self.search_angle(sin, cos)
        return angle        

@dataclass
class OpenPoseSkeleton(Skeleton):
    def __init__(self):
        super().__init__(num_keypoints=18)
        self.keypoint_names = [
            "Nose",
            "Neck",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LShoulder",
            "LElbow",
            "LWrist",
            "RHip",
            "RKnee",
            "RAnkle",
            "LHip",
            "LKnee",
            "LAnkle",
            "REye",
            "LEye",
            "REar",
            "LEar",
        ]
        self.pairs = [
            [1, 2],
            [1, 5],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [1, 11],
            [11, 12],
            [12, 13],
            [1, 0],
            [0, 14],
            [14, 16],
            [0, 15],
            [15, 17],
            [2, 17],
            [5, 16],
        ]
