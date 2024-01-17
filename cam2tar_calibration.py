import os
import cv2
import xlrd2
from math import *
import numpy as np
import pickle

Rcam = []
Tcam = []
deleted_photo_indices = []
MatrixOne = np.array([[ 9.97218832e-01,  5.34126343e-02,  5.19778021e-02,
         4.25573323e+00],
       [-5.21787365e-02,  9.98329423e-01, -2.48141633e-02,
        -8.24921679e+01],
       [-5.32163590e-02,  2.20330149e-02,  9.98339905e-01,
         1.08212254e+03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
class Calibration:
    def __init__(self):
        self.K = np.array([[1.776335005510322e+03, 0.00000000e+00, 6.035127111835385e+02],
                           [0.00000000e+00, 1.775421582824803e+03, 5.002555096623507e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        self.distortion = np.array([[-0.052162270352217, 0.401964128092692, 0.0, 0.0, 0]])
        #
        # self.K = np.array([[9.087656051837500e+02, 0.00000000e+00, 4.793915803589823e+02],
        #                    [0.00000000e+00, 9.080693417521794e+02, 3.555547448627115e+02],
        #                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
        # self.distortion = np.array([[-0.004740801521676, -0.137770171929994, 0.0, 0.0, 0]])
        # self.distortion = np.array([[0.152157896932940, -0.211380369448460, 0.0, 0.0, -8.39153683]])
        self.target_x_number = 11
        self.target_y_number = 8
        self.target_cell_size = 20

    def read_template(self, directory_name):
        array_of_img = []
        imgList = os.listdir(directory_name)
        imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片名
        for count in range(0, len(imgList)):
            filename = imgList[count]
            img = cv2.imread(directory_name + "/" + filename)  # 根据图片名读入图片
            array_of_img.append(img)
        return array_of_img

    def angle2rotation(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    def gripper2base(self, x, y, z, tx, ty, tz):
        thetaX = x / 180 * pi
        thetaY = y / 180 * pi
        thetaZ = z / 180 * pi
        R_gripper2base = self.angle2rotation(thetaX, thetaY, thetaZ)
        T_gripper2base = np.array([[tx], [ty], [tz]])
        Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
        Matrix_gripper2base = np.row_stack((Matrix_gripper2base, np.array([0, 0, 0, 1])))
        R_gripper2base = Matrix_gripper2base[:3, :3]
        T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
        return R_gripper2base, T_gripper2base


    def target2camera(self, img, img_path, f):
        global deleted_photo_indices
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.target_x_number, self.target_y_number), None)
        if not ret:
            # 未找到标定板，删除照片
            deleted_photo_indices.append(f+1) # 被删除照片的编号
            os.remove(img_path + "/" + f'{f+1}.jpg')
            return
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:, i] = corners[i, 0, :]
        object_points = np.zeros((3, self.target_x_number * self.target_y_number), dtype=np.float64)
        count = 0
        for i in range(self.target_y_number):
            for j in range(self.target_x_number):
                object_points[:2, count] = np.array(
                    # [(self.target_x_number - j - 1) * self.target_cell_size,
                    #  (self.target_y_number - i - 1) * self.target_cell_size])
                    [(j) * self.target_cell_size,
                     (i) * self.target_cell_size])
                count += 1
        retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, self.K, distCoeffs=self.distortion)
        if not retval:
            # 未能匹配，删除照片
            deleted_photo_indices.append(f+1) # 被删除照片的编号
            os.remove(img_path + "/" + f'{f+1}.jpg')
            return
        print(rvec)
        print(tvec)

        Matrix_target2camera = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec/1000))
        # print(Matrix_target2camera)
        Matrix_target2camera = np.row_stack((Matrix_target2camera, np.array([0, 0, 0, 1])))
        return Matrix_target2camera

    # def process(self, img_path, pose_path):
    def process(self, img_path):
        global deleted_photo_indices
        image_list = self.read_template(img_path)
        Hcs = []
        Hcijs = []
        # for img_path in image_list:
        for f in range(0, len(image_list)):
            img = image_list[f]
            Matrix_target2camera = self.target2camera(img, img_path, f)
            print(Matrix_target2camera)
            if Matrix_target2camera is not None:
                Hcs.append(Matrix_target2camera)
            else:
                Hcs.append(MatrixOne)

        for q in range(0, len(Hcs)-1):
            Hcij = np.dot(Hcs[q], np.linalg.inv(Hcs[q+1]))
            # Hcij = np.dot(np.linalg.inv(Hcs[q+1]), Hcs[q])
            Hcijs.append(Hcij)

        # Rcam2 = [arr.flatten() for arr in Rcam]
        # print(Rcam2)
        with open('D:/CAMERA-IMU/DATA/imu.pkl', 'rb') as f:
            imudata_array = pickle.load(f)

        for num in deleted_photo_indices:  # 删除部分帧间位姿，用0代替
            if num - 2 >= 0:
                Hcijs[num - 2] = 0
            Hcijs[num - 1] = 0

        for num in deleted_photo_indices:  # 删除部分帧间位姿，用0代替
            if num - 2 >= 0:
                imudata_array[num - 2] = 0
            imudata_array[num - 1] = 0

        Hcijs = [x for x in Hcijs if x is not 0]  # 保留可以参与计算的帧间位姿
        imudata_array = [x for x in imudata_array if x is not 0]  # 保留可以参与计算的帧间位姿

        with open('D:/CAMERA-IMU/DATA/cam.pkl', 'wb') as f:
            pickle.dump(Hcijs, f)
        with open('D:/CAMERA-IMU/DATA/imu.pkl', 'wb') as f:
            pickle.dump(imudata_array, f)
        # deleted_photo_indices = deleted_photo_indices
        # R_gripper2base_list = []
        # T_gripper2base_list = []
        # data = xlrd2.open_workbook(pose_path)
        # table = data.sheets()[0]
        # for row in range(table.nrows):
        #     x = table.cell_value(row, 0)
        #     y = table.cell_value(row, 1)
        #     z = table.cell_value(row, 2)
        #     tx = table.cell_value(row, 3)
        #     ty = table.cell_value(row, 4)
        #     tz = table.cell_value(row, 5)
        #     R_gripper2base, T_gripper2base = self.gripper2base(x, y, z, tx, ty, tz)
        #     R_gripper2base_list.append(R_gripper2base)
        #     T_gripper2base_list.append(T_gripper2base)
        # R_camera2base, T_camera2base = cv2.calibrateHandEye(R_gripper2base_list, T_gripper2base_list,
        #                                                     R_target2camera_list, T_target2camera_list)
        # return R_camera2base, T_camera2base, R_gripper2base_list, T_gripper2base_list, R_target2camera_list, T_target2camera_list
        return Hcijs, deleted_photo_indices




