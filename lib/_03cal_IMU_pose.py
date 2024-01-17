# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/7/31 18:33
"""
import numpy as np
from scipy.spatial.transform import Rotation
from ICP import *
from RotationFrom2Vectors import *
import transforms3d as tfs
from GxSingleCamMono import *
import pickle

def cal_IMU_pose_accmag(q_SD, q_IMU_pose):
	while True:
		sys_time, chip_time, acc, tempture, gyro, angle, mag = q_SD.get()
		R_imu = acc2q(acc)# 计算IMU的初始姿态
		# R_imu = east_north_up(acc, mag)  # 计算IMU的初始姿态,世界坐标系为【东北天坐标系】
		q_IMU_pose.put([R_imu, acc])


def cal_IMU_pose_gryo(q_SD, q_IMU_pose, image_array):
	init_cal = True  # 标志着第一次开始记录数据
	positionW_all_no_err = None
	imudata_array = []
	j = 1
	while True:
		# numpy_image = image_array.get()
		sys_time, chip_time, acc, tempture, gyro, angle, mag = q_SD.get()
		if init_cal == True:  # 第一次记录数据
			init_cal = False
			# 初始R为单位矩阵，加速度、速度、位置全是0
			Q_imu = np.array([1, 0, 0, 0.])  # 单位矩阵
			# R_imu = acc2q(acc)
			# Q_imu = Rotation.from_matrix(R_imu).as_quat()[[3, 0, 1, 2]]
			# R_imu = np.eye(3)
			acc_without_G = np.array([0, 0, 0.])  # 需要前后积分
			acc_without_G_all = np.array([[0, 0, 0.]])
			speedW = np.array([0, 0, 0.])  # 需要前后积分
			speedW_all = np.array([[0, 0, 0.]])
			positionW = np.array([0, 0, 0.])  # 直接累积就行
			positionW_all = np.array([[0, 0, 0.]])
			dt_all = [None, ]
		else:
			dt = sys_time - time_ex  # 记录前后两次采集数据的时间间隔dt
			# %% 计算IMU姿态
			# 方法1:使用RK4积分计算累计位姿，更加准确一些，需要按弧度计算
			Q_imu = attitude_update_RK4(Q_imu, dt, gyro_ex * np.pi / 180, gyro * np.pi / 180)  # RK4对陀螺仪进行积分，实时计算IMU在世界坐标系下的位姿
			# R_imu = Rotation.from_quat(Q_imu[[1, 2, 3, 0]]).as_matrix().T
			R_imu = Rotation.from_quat(Q_imu[[1, 2, 3, 0]]).as_matrix().T
			RR_imu = Rotation.from_quat(Q_imu[[1, 2, 3, 0]]).as_matrix()
			# # 方法2：通过欧拉角线性积分得到位姿
			# U = dt * (-gyro)
			# R = Rotation.from_euler("xyz", U, degrees=True).as_matrix()     # 直接按角度计算
			# RRR_imu = R.dot(RRR_imu)
			# RRRR_imu = RRR_imu.T

			# %% 计算IMU位置
			# 将加速度方向转换到世界坐标系下，减去重力加速度后，得到的就是各个轴向上的真实加速度
			accW = R_imu.T.dot(acc * 9.9)
			acc_without_G = accW - np.array([0, 0, 9.9])
			acc_without_G_all = np.vstack((acc_without_G_all, acc_without_G))
			speedW = speedW_ex + (acc_without_G_ex + acc_without_G) * dt / 2  # 去除重力加速度后，在世界坐标系下，前后两次加速度均值对时间积分得到速度，并对速度进行累加
			speedW_all = np.vstack((speedW_all, speedW))
			positionW = positionW + (speedW_ex + speedW) * dt / 2  # 通过对速度进行积分计算相机在世界坐标系下的位置
			positionW_all = np.vstack((positionW_all, positionW))
			dt_all.append(dt)
			# if abs(np.linalg.norm(acc) - 1) < 0.004 and np.linalg.norm(gyro) < 1.:  # 加速度小，说明IMU静止，速度和位置清零，重新计算
			if abs(np.linalg.norm(acc) - 1) < 0.01 and np.linalg.norm(gyro) < 1.:  # 加速度小，说明IMU静止，速度和位置清零，重新计算
				init_cal = True
				if speedW_all.shape[0] > 20:  # 采样点足够多的话，运动才有意义，下有必要修正位置信息
					# %% 修正速度误差
					# 利用运动终止速度为0这一特性，对上一次运动的速度和位置进行误差补偿
					speedError = speedW_all[-1, :] / speedW_all.shape[0]  # 运动终点速度/采样点数
					speedW_all_no_err = np.zeros_like(speedW_all)
					for i in range(1, speedW_all.shape[0]):
						speedW_all_no_err[i, :] = speedW_all_no_err[i - 1, :] + (acc_without_G_all[i - 1, :] + acc_without_G_all[i, :]) * dt_all[i] / 2 - speedError  # 重新计算运动区间内的速度
					# 利用修正后的速度，重新计算位置
					positionW_all_no_err = np.zeros_like(positionW_all)
					for i in range(1, positionW_all.shape[0]):
						positionW_all_no_err[i, :] = positionW_all_no_err[i - 1, :] + (speedW_all_no_err[i - 1, :] + speedW_all_no_err[i, :]) * dt_all[i] / 2  # 通过对速度进行积分计算相机在世界坐标系下的位置
						if i == positionW_all.shape[0] - 1 and Q_imu[0] < 0.997:
							# print('POSITION:',positionW_all_no_err[i, :])
							# print('ROTATION:\n',R_imu)
							# numpy_image = image_array.get()
							print(j)
							aaa = np.array(image_array)  # 在静止的时候取当前的相片信息
							numpy_image = aaa
							output_directory = 'D:/CAMERA-IMU/PHOTO/'
							filename = f"{output_directory}{j+1}.jpg"
							cv2.imwrite(filename, numpy_image)
							j = j + 1
							rmat = tfs.affines.compose(np.squeeze(positionW_all_no_err[i, :]), RR_imu, [1, 1, 1])
							imudata_array.append(rmat)
							print('TRANSLATION:\n', rmat)
							if j % 5 == 0:    # 每5个数据存储一次
								with open('D:/CAMERA-IMU/DATA/imu.pkl', 'wb') as f:
									pickle.dump(imudata_array, f)
			q_IMU_pose.put([R_imu, positionW, positionW_all, positionW_all_no_err])

		acc_without_G_ex = acc_without_G
		speedW_ex = speedW
		time_ex = sys_time
		gyro_ex = gyro


def east_north_up(acc, mag):
	'''
	IMU静止状态下，根据加速度计、磁力计，计算IMU在【东北天】坐标系下的位姿信息
	ENU坐标系，East north up
	返回的R是，R_world2imu
	:return:
	'''
	# 先由【重力加速度方向×磁力方向】，得到为X轴，即东方
	vX = np.cross(mag, acc)
	vX = vX / np.linalg.norm(vX)
	# 再由【重力加速度方向×X轴方向】，得到为Y轴，即北方
	vY = np.cross(acc, vX)
	vY = vY / np.linalg.norm(vY)

	# %% CXQ 我自己计算IMU坐标系的方法，与Zhangxin的方法结果一样
	vZ = np.cross(vX, vY)
	vZ = vZ / np.linalg.norm(vZ)
	points1 = np.array([[1, 0, 0],  # 世界坐标系（即东北天坐标系）下，xyz轴上的单位向量
	                    [0, 1, 0],
	                    [0, 0, 1.]])
	points2 = np.vstack((vX, vY, vZ))  # IMU坐标系的xyz轴单位向量 在世界坐标系下的的坐标
	_, R, _ = best_fit_transform(points1, points2)  # 计算IMU坐标系与世界坐标系之间的旋转矩阵

	# # %% Zhangxin计算IMU坐标系的方法，感觉有点麻烦，没研究  （我的结果和他的结果貌似差一个转置）
	# qX = qUtoV(vX, np.array([1, 0, 0]))         # 计算两个向量之间的四元数
	# y = qMultiVec(vY, qX)       # 向量与四元数的乘法
	# qY = qUtoV(y, np.array([0, 1, 0]))
	# qx = np.array([-qX[0], qX[1], qX[2], qX[3]])
	# qy = np.array([-qY[0], qY[1], qY[2], qY[3]])
	# q = qMultiQ(qx, qy)         # 四元数乘法
	# q = np.array([q[0], -q[1], -q[2], -q[3]])
	# if q[0] < 0:
	# 	q = -q
	# R = quatern2rotMat(q)
	return R


def attitude_update_RK4(Qk, dt, gyro0, gyro1):
	q_1 = Qk
	k1 = (1 / 2) * omegaMatrix(gyro0) @ q_1
	q_2 = Qk + dt * (1 / 2) * k1
	k2 = (1 / 2) * omegaMatrix((1 / 2) * (gyro0 + gyro1)) @ q_2
	q_3 = Qk + dt * (1 / 2) * k2
	k3 = (1 / 2) * omegaMatrix((1 / 2) * (gyro0 + gyro1)) @ q_3
	q_4 = Qk + dt * k3
	k4 = (1 / 2) * omegaMatrix(gyro1) @ q_4
	Qk_plus1 = Qk + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
	Qk_plus1 = Qk_plus1 / np.linalg.norm(Qk_plus1)

	if Qk_plus1[0] < 0:
		Qk_plus1 = -Qk_plus1

	return Qk_plus1


def omegaMatrix(omega):
	Omega = np.array([
		[0, -omega[0], -omega[1], -omega[2]],
		[omega[0], 0, omega[2], -omega[1]],
		[omega[1], -omega[2], 0, omega[0]],
		[omega[2], omega[1], -omega[0], 0]
	])

	return Omega

def acc2q(data):
	vec_IMU_z = np.array([0, 0, 1])
	vec_acc = data
	R_imu = rotation(vec_acc, vec_IMU_z)
	# q = Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
	return R_imu