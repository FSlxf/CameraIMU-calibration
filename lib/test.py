import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import pickle



# Timu2cam = np.array([[0.9998969, 0.00877812, 0.01136343, 0.15],   # 相机-IMU估测位姿
#                     [-0.04725428, -0.03615884, 0.99928508, 0.0339013],
#                     [0.00918273, -0.9993075, -0.03605821, 0.02411404],
#                     [0, 0, 0, 1]])
#
# imudata_array = np.array([[ 0.96013433,  0.27938461,  0.00929051, -0.21807959],
#        [-0.2793269 ,  0.96017023, -0.00704396,  0.10616567],
#        [-0.01088844,  0.00416806,  0.99993203, -0.02127734],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])
#
# # imuguessdata = Timu2cam @ camdata_array @ np.linalg.inv(Timu2cam)
# camguessdata = np.linalg.inv(Timu2cam) @ imudata_array @ Timu2cam
# print(camguessdata)