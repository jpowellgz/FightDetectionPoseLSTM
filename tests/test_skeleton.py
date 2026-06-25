import math
import numpy as np
from unittest import TestCase
from fight_detection_pose_lstm.skeletons import AngleCalculator, Skeleton
from fight_detection_pose_lstm.logging import logger

class TestSkeletons(TestCase):
    def test_open_pose_angle_calculation(self):
        bins = [4, 10, 15, 20]
        for bin_num in bins:
            pairs= [
                [1, 2],
                [1, 5],
                [2, 3],
                [3, 4],
                [5, 6],
                [6, 7],
                [1, 8],
                [8, 9],
            ]
            skeleton = Skeleton(num_keypoints=18, keypoints_np=np.zeros((18, 2)), pairs=pairs, fight_pairs_indexes=list(range(5)))
            calculator = AngleCalculator(angle_bins=bin_num)
            angles = calculator.angles
            total = len(angles)
            for i in range(total):
                angle = angles[i]
                sin = math.sin(angle*2*math.pi)
                cos = math.cos(angle*2*math.pi)
                skeleton.set_keypoint(1, x=0, y=0)
                skeleton.set_keypoint(2, x=cos, y=sin)
                keypoint_one, keypoint_two = skeleton.get_keypoint_pair(0)
                limb_angle_idx = calculator.calculate_limb_angle_idx(keypoint_one, keypoint_two)
                limb_angle = angles[limb_angle_idx]
                logger.info(f"angle {angle}, predicted {limb_angle}")
                self.assertIn(limb_angle, [angles[(i-1)%total], angles[i], angles[(i+1)%total]])
                