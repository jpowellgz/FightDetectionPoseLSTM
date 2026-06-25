import math
from unittest import TestCase
from fight_detection_pose_lstm.skeletons import AngleCalculator, OpenPoseSkeleton

class TestSkeletons(TestCase):
    def test_open_pose_angle_calculation(self):
        bins = [4, 10, 15, 20]
        for bin_num in bins:
            skeleton = OpenPoseSkeleton()
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
                limb_angle = calculator.calculate_limb_angle(keypoint_one, keypoint_two)
                self.assertIn(limb_angle, [angles[(i-1)%total], angles[i], angles[(i+1)%total]])
                