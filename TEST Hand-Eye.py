import pickle
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.linear_model import RANSACRegressor
import random

# 计算skew对称矩阵
def skew(A):
    return np.array([[0, -A[2][0], A[1][0]],
                     [A[2][0], 0, -A[0][0]],
                     [-A[1][0], A[0][0], 0]], dtype=np.float64)

# 实现Tsai的Hand-Eye标定算法
def Tsai_HandEye(Hcg, Hgij, Hcij):
    nStatus = len(Hgij)
    Rcg = np.eye(3) # 初始化旋转矩阵为单位矩阵
    Tcg = np.zeros((3, 1)) # 初始化平移向量为零向量

    for i in range(nStatus):
        Rgij = Hgij[i][:3, :3] # 提取Hgij中的旋转部分
        Rcij = Hcij[i][:3, :3] # 提取Hcij中的旋转部分

        rgij, _ = cv2.Rodrigues(Rgij) # 将旋转矩阵转换为旋转向量
        rcij, _ = cv2.Rodrigues(Rcij) # 将旋转矩阵转换为旋转向量

        theta_gij = np.linalg.norm(rgij) # 计算旋转向量的范数
        theta_cij = np.linalg.norm(rcij) # 计算旋转向量的范数

        rngij = rgij / theta_gij # 归一化旋转向量
        rncij = rcij / theta_cij # 归一化旋转向量

        Pgij = 2 * np.sin(theta_gij / 2) * rngij # 计算旋转向量到四元数的映射
        Pcij = 2 * np.sin(theta_cij / 2) * rncij # 计算旋转向量到四元数的映射
        tempA = skew(Pgij + Pcij) # 计算skew对称矩阵
        tempb = Pcij - Pgij

        if i == 0:
            A = tempA
            b = tempb
        else:
            A = np.vstack((A, tempA)) # 将矩阵堆叠起来
            b = np.vstack((b, tempb))

    pinA = np.linalg.pinv(A) # 计算A的伪逆
    Pcg_prime = pinA.dot(b) # 计算Pcg的中间变量
    Pcg = 2 * Pcg_prime / np.sqrt(1 + np.linalg.norm(Pcg_prime) ** 2) # 计算旋转四元数
    PcgTrs = Pcg.T
    Rcg = (1 - np.linalg.norm(Pcg) ** 2 / 2) * np.eye(3) + 0.5 * (Pcg.dot(PcgTrs) + np.sqrt(4 - np.linalg.norm(Pcg) ** 2) * skew(Pcg)) # 计算旋转矩阵
    for i in range(nStatus):
        Rgij = Hgij[i][:3, :3] # 提取Hgij中的旋转部分
        Rcij = Hcij[i][:3, :3] # 提取Hcij中的旋转部分
        Tgij = Hgij[i][:3, 3:4] # 提取Hgij中的平移部分
        Tcij = Hcij[i][:3, 3:4] # 提取Hcij中的平移部分

        tempAA = Rgij - np.eye(3) # 计算旋转矩阵与单位矩阵之差
        tempbb = Rcg.dot(Tcij) - Tgij # 计算平移向量差

        if i == 0:
            AA = tempAA
            bb = tempbb
        else:
            AA = np.vstack((AA, tempAA)) # 将矩阵堆叠起来
            bb = np.vstack((bb, tempbb))

    pinAA = np.linalg.pinv(AA) # 计算AA的伪逆
    Tcg = pinAA.dot(bb) # 计算平移向量

    Hcg[:3, :3] = Rcg
    Hcg[:3, 3:4] = Tcg
    Hcg[3, :] = [0.0, 0.0, 0.0, 1.0] # 设置变换矩阵的最后一行

with open('D:/CAMERA-IMU/DATA10/cam.pkl', 'rb') as f:
    camdata_array = pickle.load(f)
with open('D:/CAMERA-IMU/DATA10/imu.pkl', 'rb') as f:
    imudata_array = pickle.load(f)

cam_array1 = [arr for arr in camdata_array if not (arr == np.zeros((4, 4))).all()]  # 保留可以参与计算的帧间位姿
imu_array1 = [arr for arr in imudata_array if not (arr == np.zeros((4, 4))).all()]  # 保留可以参与计算的帧间位姿
print(len(cam_array1))
calibraresult1 = []
calibraresult2 = []


for ww in range(1, 30):
    # 生成随机索引
    all_indices = range(len(cam_array1))  # 假设所有子数组的长度相同
    selected_indices = random.sample(all_indices, ww*20)
    # 从每个子数组中选择随机索引位置的元素
    camdata_array = [cam_array1[i] for i in selected_indices]
    # 从每个子数组中选择随机索引位置的元素
    imudata_array = [imu_array1[i] for i in selected_indices]

    print(ww)
    print(len(camdata_array))
    Hcg = np.zeros((4, 4))
    d = 0.1
    Tsai_HandEye(Hcg, camdata_array, imudata_array)
    # print(len(cam_array))
    # print(f'初始计算外参为：{Hcg}')
    Hcg = np.array([[Hcg[0][0], Hcg[0][1], Hcg[0][2], 0],   # 相机-IMU估测位姿
                        [Hcg[1][0], Hcg[1][1], Hcg[1][2], 0],
                        [Hcg[2][0], Hcg[2][1], Hcg[2][2], 0],
                        [0, 0, 0, 1]])


    # RANSAC参数
    ransac_iterations = 50  # RANSAC迭代次数
    ransac_min_samples = ww * 20 // 2  # 拟合模型所需的最小样本数
    ransac_residual_threshold = 0.2  # 内点/外点确定的阈值

    best_Hcg = None
    best_inliers = 0
    ii = 0
    AA = 0
    diff_array = np.array([])
    for _ in range(ransac_iterations):
        # 随机选择一部分数据
        random_indices = np.random.choice(len(imudata_array), ransac_min_samples, replace=False)
        sampled_Hgij = [imudata_array[i] for i in random_indices]
        sampled_Hcij = [camdata_array[i] for i in random_indices]

        # 使用随机选择的数据调用Tsai_HandEye函数
        Hcg_estimate = np.eye(4)  # 使用单位矩阵初始化
        Tsai_HandEye(Hcg_estimate, sampled_Hcij, sampled_Hgij)

        # 比较估计的Hcg与所有数据点，计算内点数量
        inliers = 0
        for i in range(len(imudata_array)):
            # 计算估计的Hcg与数据之间的误差
            # error = np.linalg.norm(np.dot(np.linalg.inv(Hcg_estimate), imudata_array[i]) - np.dot(camdata_array[i], np.linalg.inv(Hcg_estimate)))

            imuguessdata = Hcg_estimate @ camdata_array[i] @ np.linalg.inv(Hcg_estimate)  # 根据相机帧间位姿估测imu帧间位姿
            error = np.linalg.norm(imudata_array[i][:3, 3:4] - imuguessdata[:3, 3:4]) # 判断imu位移估测值和实际值的相差多与少
            # diff_array = np.append(diff_array, diff.flatten())
            # print(error)
            if error < ransac_residual_threshold:
                inliers += 1

        # # 将 diff_array 转换为 Pandas DataFrame
        # df_diff = pd.DataFrame(diff_array.reshape(-1, 3), columns=['X', 'Y', 'Z'])
        # # 将 DataFrame 写入 Excel 文件
        # df_diff.to_excel('diff_output.xlsx', index=False)

        # 如果本次迭代具有更多的内点，则更新最佳结果
        if inliers >= best_inliers:
            best_inliers = inliers
            best_Hcg = Hcg_estimate
            AA = AA + best_Hcg[:3, 3:4]
            ii = ii + 1
            BB = AA / ii
            # print(best_inliers)
            # print(BB)
    print("最佳的外参估计，内点数量为", best_inliers, "个:")
    print(BB)
    calibraresult1 = np.append(calibraresult1, BB)
print(calibraresult1)




with open('D:/CAMERA-IMU/DATA15/cam.pkl', 'rb') as f:
    camdata_array = pickle.load(f)
with open('D:/CAMERA-IMU/DATA15/imu.pkl', 'rb') as f:
    imudata_array = pickle.load(f)

cam_array1 = [arr for arr in camdata_array if not (arr == np.zeros((4, 4))).all()]  # 保留可以参与计算的帧间位姿
imu_array1 = [arr for arr in imudata_array if not (arr == np.zeros((4, 4))).all()]  # 保留可以参与计算的帧间位姿
print(len(cam_array1))

for ww in range(1, 30):
    # 生成随机索引
    all_indices = range(len(cam_array1))  # 假设所有子数组的长度相同
    selected_indices = random.sample(all_indices, ww*20)
    # 从每个子数组中选择随机索引位置的元素
    camdata_array = [cam_array1[i] for i in selected_indices]
    # 从每个子数组中选择随机索引位置的元素
    imudata_array = [imu_array1[i] for i in selected_indices]

    print(ww)
    print(len(camdata_array))
    Hcg = np.zeros((4, 4))
    d = 0.1
    Tsai_HandEye(Hcg, camdata_array, imudata_array)
    # print(len(cam_array))
    # print(f'初始计算外参为：{Hcg}')
    Hcg = np.array([[Hcg[0][0], Hcg[0][1], Hcg[0][2], 0],   # 相机-IMU估测位姿
                        [Hcg[1][0], Hcg[1][1], Hcg[1][2], 0],
                        [Hcg[2][0], Hcg[2][1], Hcg[2][2], 0],
                        [0, 0, 0, 1]])


    # RANSAC参数
    ransac_iterations = 50  # RANSAC迭代次数
    ransac_min_samples = ww * 20 // 2  # 拟合模型所需的最小样本数
    ransac_residual_threshold = 0.2  # 内点/外点确定的阈值

    best_Hcg = None
    best_inliers = 0
    ii = 0
    AA = 0
    diff_array = np.array([])
    for _ in range(ransac_iterations):
        # 随机选择一部分数据
        random_indices = np.random.choice(len(imudata_array), ransac_min_samples, replace=False)
        sampled_Hgij = [imudata_array[i] for i in random_indices]
        sampled_Hcij = [camdata_array[i] for i in random_indices]

        # 使用随机选择的数据调用Tsai_HandEye函数
        Hcg_estimate = np.eye(4)  # 使用单位矩阵初始化
        Tsai_HandEye(Hcg_estimate, sampled_Hcij, sampled_Hgij)

        # 比较估计的Hcg与所有数据点，计算内点数量
        inliers = 0
        for i in range(len(imudata_array)):
            # 计算估计的Hcg与数据之间的误差
            # error = np.linalg.norm(np.dot(np.linalg.inv(Hcg_estimate), imudata_array[i]) - np.dot(camdata_array[i], np.linalg.inv(Hcg_estimate)))

            imuguessdata = Hcg_estimate @ camdata_array[i] @ np.linalg.inv(Hcg_estimate)  # 根据相机帧间位姿估测imu帧间位姿
            error = np.linalg.norm(imudata_array[i][:3, 3:4] - imuguessdata[:3, 3:4]) # 判断imu位移估测值和实际值的相差多与少
            # diff_array = np.append(diff_array, diff.flatten())
            # print(error)
            if error < ransac_residual_threshold:
                inliers += 1

        # # 将 diff_array 转换为 Pandas DataFrame
        # df_diff = pd.DataFrame(diff_array.reshape(-1, 3), columns=['X', 'Y', 'Z'])
        # # 将 DataFrame 写入 Excel 文件
        # df_diff.to_excel('diff_output.xlsx', index=False)

        # 如果本次迭代具有更多的内点，则更新最佳结果
        if inliers >= best_inliers:
            best_inliers = inliers
            best_Hcg = Hcg_estimate
            AA = AA + best_Hcg[:3, 3:4]
            ii = ii + 1
            BB = AA / ii
            # print(best_inliers)
            # print(BB)
    print("最佳的外参估计，内点数量为", best_inliers, "个:")
    print(BB)
    calibraresult2 = np.append(calibraresult2, BB)
print(calibraresult2)
calibraresult = calibraresult2 - calibraresult1

# 使用切片每隔3个元素取一个
calibraresultX = calibraresult[::3]
calibraresultY = calibraresult[1::3]
calibraresultZ = calibraresult[2::3]

calibraresultX = (calibraresultX - 0.05) / 1.732050 * 1000
calibraresultY = (calibraresultY) / 1.732050 * 1000
calibraresultZ = (calibraresultZ) / 1.732050 * 1000

# 将NumPy数组转换为DataFrame
df = pd.DataFrame({
    'Column1': calibraresultX,
    'Column2': calibraresultY,
    'Column3': calibraresultZ
})

# 保存DataFrame为Excel文件
df.to_excel("calibraresult.xlsx", index=False)

print(calibraresultX)