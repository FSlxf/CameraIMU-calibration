# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/7/25 19:17
"""
import cv2, time
import gxipy as gx
from computer_vision import *

def blob_init():
	# 有时候圆点靶标检测不到，可能是因为blob检测有问题
	blobParams = cv2.SimpleBlobDetector_Params()
	# Filter by Circularity   圆度过滤    4pi*面积/周长的平方      区分多边形和圆
	blobParams.filterByCircularity = True
	blobParams.minCircularity = 0.5
	blobParams.filterByArea = True
	blobParams.minArea = 200
	blobParams.maxArea = 5000
	blobDetector = cv2.SimpleBlobDetector_create(blobParams)
	return blobDetector

def get_image():
	cam.TriggerSoftware.send_command()
	RawImg = cam.data_stream[0].get_image()
	if RawImg is None:
		print("Getting image failed!!")
		return False, None, None
	img_gray = RawImg.get_numpy_array()
	img_gray = cv2.undistort(img_gray, K, dist)
	img_gray = cv2.resize(img_gray,(img_gray.shape[1]//2,img_gray.shape[0]//2))
	img_gray = cv2.equalizeHist(img_gray)
	img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
	return True, img_gray, img_rgb


# create a device manager
DeviceManager = gx.DeviceManager()
DevNum, DevInfo = DeviceManager.update_device_list()    # DevNum:设备数量，DevInfo：设备信息列表
if DevNum == 0:
	print("Number of enumerated devices is 0")
	exit()
cam = DeviceManager.open_device_by_index(1) # 打开第1个设备，没有0

# 设置软触发
cam.TriggerMode.set(gx.GxSwitchEntry.ON)
cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
cam.ExposureTime.set(30000) # 曝光
cam.Gain.set(0.0)          # 增益



