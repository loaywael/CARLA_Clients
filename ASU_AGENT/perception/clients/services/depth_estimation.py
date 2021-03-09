import matplotlib.pyplot as plt
import numpy as np
import cv2 


class DepthEstimator:
    def __init__(self, left_matrix, right_matrix):
        self.__lMatrix = left_matrix
        self.__rMatrix = right_matrix
        self.__lK, self.__lR, self.__lT = self.decompose_projection(self.__lMatrix)
        self.__rK, self.__rR, self.__rT = self.decompose_projection(self.__rMatrix)
        self.__f = self.lK[0, 0]
        self.__b = abs(self.rT[1] - self.lT[1])
        self.__stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=6*16,
            blockSize=11,
            P1=8*3*(11**2), 
            P2=32*3*(11**2), 
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY
        )

    @staticmethod
    def decompose_projection(Matrix):
        K, R, t, _, _ = cv2.cv2.decomposeProjectionMatrix(Matrix)
        return K, R, t/t[3]

    @staticmethod
    def comp_disparity(left, right):
        desparity_map = self.__stereo_matcher.compute(left, right)
        return desparity_map.astype("float32") / 16

    def __call__(self, left, right):
        disparity_map = comp_disparity(left, right)
        disparity_map[disparity_map == 0 or disparity_map ==-1] = 0.01
        depthmap = f * b / disparity_map
        return depthmap
