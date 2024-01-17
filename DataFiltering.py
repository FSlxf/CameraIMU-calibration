import pickle
from cam2tar_calibration import *


image_path = "D:/CAMERA-IMU/PHOTO"
calibrator = Calibration()
Hcijs, deleted_photo_indices = calibrator.process(image_path)  # 计算相机帧间位姿

print("deleted_photo_indices:",deleted_photo_indices)

with open('D:/CAMERA-IMU/DATA/cam.pkl', 'rb') as f:
    camdata_array = pickle.load(f)
with open('D:/CAMERA-IMU/DATA/imu.pkl', 'rb') as f:
    imudata_array = pickle.load(f)

# for num in deleted_photo_indices:  # 删除部分帧间位姿，用0代替
#     if num - 2 >= 0:
#         camdata_array[num - 2] = 0
#     camdata_array[num - 1] = 0
#
# for num in deleted_photo_indices:  # 删除部分帧间位姿，用0代替
#     if num - 2 >= 0:
#         imudata_array[num - 2] = 0
#     imudata_array[num - 1] = 0

# camdata_array = [x for x in camdata_array if x is not 0] # 保留可以参与计算的帧间位姿
# imudata_array = [x for x in imudata_array if x is not 0] # 保留可以参与计算的帧间位姿


# Timu2cam = np.array([[0.9998969, 0.00877812, 0.01136343, 0.15],   # 相机-IMU估测位姿
#                     [-0.04725428, -0.03615884, 0.99928508, 0.025],
#                     [0.00918273, -0.9993075, -0.03605821, 0],
#                     [0, 0, 0, 1]])
#
# for i in range(len(camdata_array)):
#     imuguessdata = Timu2cam @ camdata_array[i] @ np.linalg.inv(Timu2cam)  # 根据相机帧间位姿估测imu帧间位姿
#     # 计算A和B的差值
#     diff = np.abs(imudata_array[i][:3, 3:4] - imuguessdata[:3, 3:4]) # 判断imu位移估测值和实际值的相差多与少
#     print("imuguessdata:",imuguessdata)
#     print("imudata_array:",imudata_array[i])
#     print("diff:",diff)
#     # 判断差值是否在正负3之内
#     result = (diff[0] <= 0.05 and diff[1] <= 0.05 and diff[2] <= 0.05) # 如果x、y、z轴的位移都相差5cm以内，则保留，否则删除
#     if result:
#         continue
#     else:
#         camdata_array[i] = 0
#         imudata_array[i] = 0
#
# camdata_array = [x for x in camdata_array if not (x == np.zeros((4, 4))).all()] # 保留可以参与计算的帧间位姿
# imudata_array = [x for x in imudata_array if not (x == np.zeros((4, 4))).all()] # 保留可以参与计算的帧间位姿
# print("camdata_array:",camdata_array)
# print("imudata_array:",imudata_array)
# with open('D:/CAMERA-IMU/imudata_array.pkl', 'wb') as f:
#     pickle.dump(imudata_array, f)
# with open('D:/CAMERA-IMU/camdata_array.pkl', 'wb') as f:
#     pickle.dump(camdata_array, f)
