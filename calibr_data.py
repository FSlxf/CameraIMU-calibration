import pickle
import numpy as np
import os

# 指定文件夹路径
folder_path = "D:\CAMERA-IMU\github\calibra_data"  # 请将路径替换为实际文件夹路径
# 获取文件夹中的所有文件名
file_names = [file for file in os.listdir(folder_path) if file.startswith("cam") and file.endswith(".pkl")]
# file_names = [file for file in os.listdir(folder_path) if file.startswith("camdata_array") and file.endswith(".pkl")]
# 根据文件名排序文件列表
file_names.sort()
# 初始化一个空的数组，用于存储连接后的数据
concatenated_data = None
# 遍历文件并连接数据
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "rb") as file:
        data = pickle.load(file)

    if concatenated_data is None:
        concatenated_data = data
    else:
        concatenated_data = np.concatenate((concatenated_data, data), axis=0)
# 现在concatenated_data中包含了所有文件中的数据连接在一起
with open('D:\CAMERA-IMU\github\DATA\cam.pkl', 'wb') as f:
    pickle.dump(concatenated_data, f)
# with open('D:/CAMERA-IMU/camdata_array.pkl', 'wb') as f:
#     pickle.dump(concatenated_data, f)



# 指定文件夹路径
folder_path = "D:\CAMERA-IMU\github\calibra_data"  # 请将路径替换为实际文件夹路径
# 获取文件夹中的所有文件名
file_names = [file for file in os.listdir(folder_path) if file.startswith("imu") and file.endswith(".pkl")]
# file_names = [file for file in os.listdir(folder_path) if file.startswith("imudata_array") and file.endswith(".pkl")]
# 根据文件名排序文件列表
file_names.sort()
# 初始化一个空的数组，用于存储连接后的数据
concatenated_data = None
# 遍历文件并连接数据
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "rb") as file:
        data = pickle.load(file)

    if concatenated_data is None:
        concatenated_data = data
    else:
        concatenated_data = np.concatenate((concatenated_data, data), axis=0)
# 现在concatenated_data中包含了所有文件中的数据连接在一起
with open('D:\CAMERA-IMU\github\DATA\imu.pkl', 'wb') as f:
    pickle.dump(concatenated_data, f)
# with open('D:/CAMERA-IMU/imudata_array.pkl', 'wb') as f:
#     pickle.dump(concatenated_data, f)