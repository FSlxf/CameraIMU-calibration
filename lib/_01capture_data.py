# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/7/31 18:09
"""
import time, serial


def open_serial(com, bps):
	# 打开串口
	try:
		ser = serial.Serial(com, bps)
	except Exception as e:
		print(e)
		exit()
	return ser


def capture_sensor_data(q_SD_raw, com_port, com_bps, print_fps = False):
	ser = open_serial(com_port, com_bps)
	start = time.perf_counter()
	SD_raw = {}  # 存储完整的一帧数据
	ser.reset_input_buffer()  # 清空接受区缓存
	# ser.reset_output_buffer()
	# %% 开始循环接收数据
	while True:
		if ser.read(1).hex() == '55':  # 0x55是每一组数据的数据头， 后边分 0x51 52 53 54等情况，这个要研究WIT私有协议
			key = ser.read(1).hex()  # 读取数据帧标志 50 51 52 53 54
			if key == "50":  # 接收到磁力数据，即一帧数据的末尾
				SD_raw['sys_time'] = time.perf_counter()  # 将接收到50的时间，作为IMU采集传感器数据的时间
			data = [int.from_bytes(ser.read(1), byteorder='big') for i in range(9)]
			SD_raw[key] = data
			if key == "54":  # 接收到最后一帧数据，
				end = time.perf_counter()
				if len(SD_raw.keys()) == 6:  # 完整数据包括的可以为：50 51 52 53 54 sys_time
					if print_fps:
						print('\n回传速率为：{0:.2f}HZ'.format(1 / (end - start)))  # 显示帧率
					q_SD_raw.put(SD_raw)
				else:
					print('Frame length wrong!!')
				start = end
				SD_raw = {}
	# ser.close()#关闭串口
