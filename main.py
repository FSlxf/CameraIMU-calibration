# -*- coding:utf-8 _*-
import multiprocessing as mp
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pyvista as pv

from lib._01capture_data import *
from lib._02parse_data import *
from lib._03cal_IMU_pose import *
from ShowPoints import *
from scipy import io
from GxSingleCamMono import *
from computer_vision import *

data = []
data2 = []

def program_exit():
	# global data2
	exit()

if __name__ == '__main__':
	print_fps = False          # 原始IMU数据采集的帧率
	print_SD = False             # 打印解析后的IMU数据

	q_SD_raw = mp.Queue(maxsize=100)  # 先进先出队列，实现不同进程数据交互，用于传递原始的IMU数据
	q_SD = mp.Queue(maxsize=100)  # 用于传递解析后的IMU数据
	q_IMU_pose = mp.Queue(maxsize=100) # 用于传递解算后的IMU位姿
	# image_array = mp.Queue(maxsize=1000) # 用于传递相机的图片
	image_array = mp.Manager().list()# 主进程与子进程共享这个数组

	# %% 相机拍摄
	p_cam_capture = mp.Process(target=CamMono, args = (image_array,), daemon=True)
	p_cam_capture.start()

	# %% 开启数据处理进程
	p_getdata = mp.Process(target=capture_sensor_data, args=(q_SD_raw, 'COM3', 115200, print_fps), daemon=True)  # 开启一个线程接收数据， 守护进程（父进程停止，则子进程也马上停止）
	p_getdata.start()

	# %% 解析IMU数据
	p_parsedata = mp.Process(target=parse_sensor_data, args=(q_SD_raw, q_SD, print_SD), daemon=True)
	p_parsedata.start()

	# %% IMU姿态解算
	# p_cal_IMU_pose = mp.Process(target=cal_IMU_pose_accmag, args=(q_SD, q_IMU_pose), daemon=True)  # 不能靠这个计算，因为加速度、磁力计确定IMU在世界坐标系的位姿，必须在IMU静止的情况下完成
	p_cal_IMU_pose = mp.Process(target=cal_IMU_pose_gryo, args=(q_SD, q_IMU_pose, image_array), daemon=True)  # 必须靠陀螺仪计算
	p_cal_IMU_pose.start()


	# p_cam_capture.join()

	p = pv.Plotter()
	# p.add_background_image('1.png') # %% 添加背景图作为第一张图
	p.add_key_event('space', program_exit)          # %% 空格键终止程序运行
	pv_add_axis_WCS(p, 0.1, colors=['r', 'g', 'b'], opacity=0.5)
	# p.show(interactive_update=True)
	pv_add_bounds_grid_pv(p, (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))   # 单位是米
	p.camera_position = [(0.2983763861344092, -2.5780296633117774, 0.7031720526912835),
	                     (0.03038858474487896, 0.32047312595145155, -0.17993760933909592),
	                     (-0.04377551040788642, 0.2874654125841464, 0.9567901239333293)]

	while True:
		try:
			# p.remove_actor(a1)
			# p.remove_actor(a2)
			# p.remove_actor(a3)
			# p.remove_actor(a4)
			p.remove_actor(a5)
		except:
			pass

		R_imu, positionW, positionW_all, positionW_all_no_err = q_IMU_pose.get()       # 获得解算的IMU在世界坐标系下的姿态
		# img_gray = image_array.get()       # 获得相机图像
		# img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
		# 显示IMU位置和姿态
		axis_points_new = np.linalg.inv(R_imu).dot(axis_WCS.T).T*0.1+positionW
		if positionW_all_no_err is not None:        # 显示上一次运动修正后的位置信息
			a5 = pv_add_points(p, positionW_all_no_err, color='r')
		# p.remove_background_image()
		# p.add_background_image(img_warp(img_rgb))
		# p.add_background_image('1.png')  # %% 添加背景图作为第一张图
		# p.update()

