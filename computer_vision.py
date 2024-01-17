# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/7/26 9:01
"""
import numpy as np
import cv2
import pyvista as pv

K = np.array([[4648.99488, 0, 1198.32903],
              [0, 4648.34338, 1013.47133],
              [0, 0, 1]])
dist = np.array([-0.02512, 0.42340, 0.00012, 0.00022, 0.00000])

objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
objp = objp/2


def compute_position(img_gray, img_rgb, blobDetector):
	ret, corners = cv2.findCirclesGrid(img_gray, (4, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector = blobDetector)
	if ret == False:
		print('Find circles failed!!')
		return False, None, None, img_rgb
	ret, Rvec, tvec = cv2.solvePnP(objp, corners*2, K, dist)
	if ret == False:
		print('Solve position failed!!')
		return False, None, None, img_rgb
	R = cv2.Rodrigues(Rvec)[0]
	cv2.drawChessboardCorners(img_rgb, (4, 11), corners, ret)
	return True, R, tvec, img_rgb

def img_warp(array):
	big_img = np.ones((3000, 4200, 3), dtype=np.uint8)*100
	big_img[0:array.shape[0], 0:array.shape[1], :] = array


	array = cv2.flip(big_img, 0)
	h, w, c = array.shape
	image_data = pv.wrap(np.zeros((w, h, 1)))       # 随便warp一个具有相同尺寸的图像
	color_scalars = array.reshape(-1, 3)           # 将原始图像的颜色，赋值给scales, 每个scale对应为一个像素的颜色
	image_data.point_data['values'] = np.array(color_scalars)
	return image_data