import numpy as np
from numpy.linalg import norm
import cv2
np.set_printoptions(suppress=True,precision=4)


def rotation(u, v):     # u,v 为(3,1)或(1,3)或(3,) R*u=v  R*X_v + t =X_u (X_v为以V向量为坐标系下的点坐标，X_u为以u向量为坐标系下的点坐标)
   '''
   用于计算同一坐标系下空间两个向量（u,v）之间的旋转矩阵R
   设X_u为u向量坐标系下的空间点三维坐标
   X_v为同一点在v向量坐标系下的三维坐标，
   则R(X_v)=X_u     R*u 即为v的方向向量
   '''
   u = u / norm(u, 2)  # 单位化
   v = v / norm(v, 2)
   w = np.cross(u.flatten(), v.flatten()).reshape(-1, 1)   # 两向量法向，即旋转轴
   s = norm(w) # 旋转轴长度
   c = u.flatten().dot(v)  # cos(theta)
   # print(np.arccos(c)*180/np.pi)     # 向量旋转角度
   C = np.array([[0, -w[2,0], w[1,0]], [w[2,0], 0, -w[0,0]], [-w[1,0], w[0,0], 0]])  # C为w的反对称矩阵
   if s==0:
      R = np.eye(3,dtype=np.float32)
   else:
      R = np.eye(3) + C + ((1 - c) / (s ** 2)) * C.dot(C)
   return R