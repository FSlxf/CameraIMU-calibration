import numpy as np
import cv2
from math import *
import math
from scipy.spatial.transform import Rotation as R
import transforms3d as tfs
import pickle

# 计算skew对称矩阵
def skew(A):
    return np.array([[0, -A[2][0], A[1][0]],
                     [A[2][0], 0, -A[0][0]],
                     [-A[1][0], A[0][0], 0]], dtype=np.float64)

def get_matrix_eular_radu(x, y, z, rx, ry, rz):
    rmat = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz))
    rmat = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), rmat, [1, 1, 1])
    # print(rmat)
    return rmat

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

with open('D:/CAMERA-IMU/DATA/cam.pkl', 'rb') as f:
    Hcijs = pickle.load(f)
with open('D:/CAMERA-IMU/DATA/imu.pkl', 'rb') as f:
    Hgs = pickle.load(f)

print(len(Hcijs))
print(len(Hgs))

Hcg = np.zeros((4, 4))
Tsai_HandEye(Hcg, Hgs, Hcijs)
# Tsai_HandEye(Hcg, Hcijs, Hgs)
print(Hcg)


