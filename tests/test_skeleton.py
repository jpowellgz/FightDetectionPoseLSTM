import math
from unittest import TestCase
from src.fightdetectionposelstm.skeletons import OpenPoseSkeleton

class TestSkeletons(TestCase):
    def test_open_pose_angle_calculation(self):
        bins = [4, 10, 15, 20]
        for bin_num in bins:
            skeleton = OpenPoseSkeleton(angle_bins=bin_num)
            angles = skeleton.angles
            total = len(angles)
            for i in range(total):
                angle = angles[i]
                sin = math.sin(angle*2*math.pi)
                cos = math.cos(angle*2*math.pi)
                skeleton.set_keypoint(1, x=0, y=0)
                skeleton.set_keypoint(2, x=cos, y=sin)
                limb_angle = skeleton.calculate_limb_angle(0)
                self.assertIn(limb_angle, [angles[(i-1)%total], angles[i], angles[(i+1)%total]])
                