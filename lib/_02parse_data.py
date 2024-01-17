# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/7/31 17:28
"""
import time
from scipy import io
import pickle
import numpy as np
import pandas as pd
# from CXQ.Basic.Timer import *

gyroRange = 2000.0  # 角速度量程
accRange = 16.0  # 加速度量程
angleRange = 180.0  # 角度量程
data = []
data2 = []

def get_chiptime(datahex):
	"""
	芯片时间结算
	:param datahex: 原始始数据包
	:return:
	"""
	tempVals = []  # 临时结算数据
	for i in range(0, 4):
		tIndex = i * 2
		tempVals.append(datahex[tIndex + 1] << 8 | datahex[tIndex])
	# year = 2000 + (tempVals[0] & 0xff)  # 年
	# moth = ((tempVals[0] >> 8) & 0xff)  # 月
	# day = (tempVals[1] & 0xff)  # 日
	hour = ((tempVals[1] >> 8) & 0xff)  # 时
	minute = (tempVals[2] & 0xff)  # 分
	second = ((tempVals[2] >> 8) & 0xff)  # 秒
	millisecond = tempVals[3]  # 毫秒
	chip_time = millisecond / 1000 + second + minute * 60 + hour * 3600
	return chip_time


def get_acc_tempture(datahex):
	"""
	加速度、温度结算
	:param datahex: 原始始数据包
	:return:
	"""
	axl = datahex[0]
	axh = datahex[1]
	ayl = datahex[2]
	ayh = datahex[3]
	azl = datahex[4]
	azh = datahex[5]

	tempVal = (datahex[7] << 8 | datahex[6])
	acc_x = (axh << 8 | axl) / 32768.0 * accRange
	acc_y = (ayh << 8 | ayl) / 32768.0 * accRange
	acc_z = (azh << 8 | azl) / 32768.0 * accRange
	if acc_x >= accRange:
		acc_x -= 2 * accRange
	if acc_y >= accRange:
		acc_y -= 2 * accRange
	if acc_z >= accRange:
		acc_z -= 2 * accRange
	acc = np.array([round(acc_x, 4), round(acc_y, 4), round(acc_z, 4)])
	tempture = round(tempVal / 100.0, 4)  # 温度结算,并保留两位小数
	return acc, tempture


def get_gyro(datahex):
	"""
	角速度结算
	"""
	wxl = datahex[0]
	wxh = datahex[1]
	wyl = datahex[2]
	wyh = datahex[3]
	wzl = datahex[4]
	wzh = datahex[5]

	gyro_x = (wxh << 8 | wxl) / 32768.0 * gyroRange
	gyro_y = (wyh << 8 | wyl) / 32768.0 * gyroRange
	gyro_z = (wzh << 8 | wzl) / 32768.0 * gyroRange
	if gyro_x >= gyroRange:
		gyro_x -= 2 * gyroRange
	if gyro_y >= gyroRange:
		gyro_y -= 2 * gyroRange
	if gyro_z >= gyroRange:
		gyro_z -= 2 * gyroRange
	gyro = np.array([round(gyro_x, 4), round(gyro_y, 4), round(gyro_z, 4)])
	return gyro


def get_angle(datahex):
	"""
	角度结算
	"""
	rxl = datahex[0]
	rxh = datahex[1]
	ryl = datahex[2]
	ryh = datahex[3]
	rzl = datahex[4]
	rzh = datahex[5]
	angle_x = (rxh << 8 | rxl) / 32768.0 * angleRange
	angle_y = (ryh << 8 | ryl) / 32768.0 * angleRange
	angle_z = (rzh << 8 | rzl) / 32768.0 * angleRange
	if angle_x >= angleRange:
		angle_x -= 2 * angleRange
	if angle_y >= angleRange:
		angle_y -= 2 * angleRange
	if angle_z >= angleRange:
		angle_z -= 2 * angleRange
	angle = np.array([round(angle_x, 3), round(angle_y, 3), round(angle_z, 3)])
	return angle


def get_mag(datahex):
	"""
	磁场结算（单位：微特），https://wit-motion.yuque.com/wumwnr/docs/qe67vq?#%20%E3%80%8AWT901C-TTL/232%E4%BA%A7%E5%93%81%E8%A7%84%E6%A0%BC%E4%B9%A6%E3%80%8B
	"""
	_x = get_int(bytes([datahex[0], datahex[1]])) * 0.00667  # 协议上显示，分辨率为0.0667mGauss/LSB,即0.00667uT/位
	_y = get_int(bytes([datahex[2], datahex[3]])) * 0.00667
	_z = get_int(bytes([datahex[4], datahex[5]])) * 0.00667
	mag = np.array([round(_x, 4), round(_y, 4), round(_z, 4)])
	return mag


def get_int(dataBytes):
	"""
	int转换有符号整形   = C# BitConverter.ToInt16
	:param dataBytes: 字节数组
	:return:
	"""
	# return -(data & 0x8000) | (data & 0x7fff)
	return int.from_bytes(dataBytes, "little", signed=True)


# @ time_consuming
def parse_sensor_data(q_SD_raw, q_SD, print_SD=False):
	# global data2, data
	while True:
		SD_raw = q_SD_raw.get()
		if (np.sum(SD_raw['50'][:-1]) + 0x55 + 0x50) & 0xff == SD_raw['50'][-1] and \
				(np.sum(SD_raw['51'][:-1]) + 0x55 + 0x51) & 0xff == SD_raw['51'][-1] and \
				(np.sum(SD_raw['52'][:-1]) + 0x55 + 0x52) & 0xff == SD_raw['52'][-1] and \
				(np.sum(SD_raw['53'][:-1]) + 0x55 + 0x53) & 0xff == SD_raw['53'][-1] and \
				(np.sum(SD_raw['54'][:-1]) + 0x55 + 0x54) & 0xff == SD_raw['54'][-1]:
			sys_time = SD_raw['sys_time']
			chip_time = get_chiptime(SD_raw['50'])
			acc, tempture = get_acc_tempture(SD_raw['51'])
			gyro = get_gyro(SD_raw['52'])
			angle = get_angle(SD_raw['53'])
			mag = get_mag(SD_raw['54'])


			# my_array = np.array(data2)
			# 将数据添加到数组
			# data.append((chip_time, acc, gyro))

			# data_dict = {'my_array': my_array}
			# filename = 'my_data.mat'
			# io.savemat(filename, data_dict)

			q_SD.put([sys_time, chip_time, acc, tempture, gyro, angle, mag])
			if print_SD:
				print('***************************')
				print('Sys Time:', sys_time)
				print('Chip Time:', chip_time)
				print('Acc:', acc)
				print('Temp:', tempture)
				print('Gyr:', gyro)
				print('Angle:', angle)
				print('Mag:', mag)

