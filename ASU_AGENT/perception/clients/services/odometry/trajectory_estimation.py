import matplotlib.pyplot as plt
import numpy as np
import cv2 


class TrajectoryEstimator:
    def __init__(self, matcher_threshold):
        self.__matcher_threshold = matcher_threshold
        self.__search_params = dict(checks=50)
        self.__index_params = dict(algorithm=1, trees=5)
        self.__descriptor = cv2.SIFT_create(500)
        self.__K = np.float32(
            [[640, 0,   640 ], 
             [0,   480, 480 ], 
             [0,   0,   1   ]]
        )
        self.__matcher = cv2.FlannBasedMatcher(
            self.__index_params, self.__search_params
        )
        self.__kpts_buffer = []
        self.__des_buffer = []
        self.__matched_kpts = []
        self.__trajectory = []
        self.__pair_buffer = []
        self.__P = np.eye(4)
        self.__P_next = self.__P.copy()
    
    @staticmethod
    def _filter_keypoints(matches, threshold):
        good_only = []
        for i in range(len(matches)):
            if len(matches[i]) == 2:
                m, n = matches[i]
                if m.distance < threshold * n.distance:
                    good_only.append(m)
        return good_only

    @staticmethod
    def _estimate_motion(matches, img1_kpts, img2_kpts, K):
        R = np.eye(3)
        t = np.zeros((1, 3))
        img1_pts = []
        img2_pts = []
        for match in matches:
            img1_pts.append(img1_kpts[match.queryIdx].pt)
            img2_pts.append(img2_kpts[match.trainIdx].pt)
        pts1, pts2 = np.array(img1_pts), np.array(img2_pts)
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        return R, t, img1_kpts, img2_kpts

    def _estimate_trajectory(self, R, t, img1_points, img2_points):
        self.__P_next[:-1, :-1] = R
        self.__P_next[:-1, -1:] = t
        P_next_inv = np.linalg.inv(self.__P_next)
        self.__P = self.__P @ P_next_inv
        cam_pose = self.__P[:3, 3]
        self.__trajectory.append(cam_pose)
        return cam_pose

    @property
    def trajectory(self):
        return np.array(self.__trajectory[:]).T

    def __call__(self, frame):
        self.__pair_buffer.append(frame)
        kpts, des = self.__descriptor.detectAndCompute(frame, None)
        self.__kpts_buffer.append(kpts)
        self.__des_buffer.append(des)
        if len(self.__pair_buffer) == 2:
            matched_pts = self.__matcher.knnMatch(*self.__des_buffer, k=2)
            matched_pts = (self._filter_keypoints(matched_pts, self.__matcher_threshold))
            self.__matched_kpts.append(matched_pts)
            R, t, img1_pts, img2_pts = self._estimate_motion(
                matched_pts, *self.__kpts_buffer, self.__K
            )
            trajectory_point = self._estimate_trajectory(R, t, img1_pts, img2_pts)
            self.__pair_buffer = []
            self.__kpts_buffer = []
            self.__des_buffer = []
            return trajectory_point



