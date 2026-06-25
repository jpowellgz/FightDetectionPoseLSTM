import numpy as np

SOBEL_LR_KERNEL = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
SOBEL_RL_KERNEL = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
AVERAGING_KERNEL = np.ones((5,5),np.float32)/25
