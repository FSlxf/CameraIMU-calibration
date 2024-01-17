# coding:utf-8
# icp模块包含两种ICP拼接算法：
# （1）
# T,R,t = best_fit_transform(A, B):
# B = R.dot(A.T) + t).T
# 最小二乘法线性解,需要A,B点云数据顺序匹配
# （2）
# icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
# 迭代法求解,不需要A,B点云数据顺序匹配

import numpy as np
import cv2


def best_fit_transform(A, B):
	'''
	最小二乘法线性解,需要A,B点云数据顺序匹配 R*A+t=B
	Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
	Input:
	  A: Nxm numpy array of corresponding points
	  B: Nxm numpy array of corresponding points
	Returns:
	  T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
	  R: mxm rotation matrix
	  t: mx1 translation vector
	'''

	assert A.shape == B.shape

	# get number of dimensions
	m = A.shape[1]

	# translate points to their centroids
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	AA = A - centroid_A
	BB = B - centroid_B

	# rotation matrix
	H = np.dot(AA.T, BB)
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T, U.T)

	# special reflection case
	if np.linalg.det(R) < 0:
		Vt[m - 1, :] *= -1
		R = np.dot(Vt.T, U.T)

	# translation
	t = centroid_B.T - np.dot(R, centroid_A.T)

	# homogeneous transformation
	T = np.identity(m + 1)  # 3*4 【R|t】
	T[:m, :m] = R
	T[:m, m] = t

	return T, R, t.reshape((-1, 1))


def nearest_neighbor(src, dst):
	'''
	Find the nearest (Euclidean) neighbor in dst for each point in src
	Input:
		src: Nxm array of points
		dst: Nxm array of points
	Output:
		distances: Euclidean distances of the nearest neighbor
		indices: dst indices of the nearest neighbor
	'''
	from sklearn.neighbors import NearestNeighbors
	assert src.shape == dst.shape
	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(dst)
	distances, indices = neigh.kneighbors(src, return_distance=True)
	return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
	'''
	迭代法求解,不需要A,B点云数据顺序匹配
	The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
	Input:
		A: Nxm numpy array of source mD points
		B: Nxm numpy array of destination mD point
		init_pose: (m+1)x(m+1) homogeneous transformation
		max_iterations: exit algorithm after max_iterations
		tolerance: convergence criteria
	Output:
		T: final homogeneous transformation that maps A on to B
		distances: Euclidean distances (errors) of the nearest neighbor
		i: number of iterations to converge
	'''

	assert A.shape == B.shape

	# get number of dimensions
	m = A.shape[1]

	# make points homogeneous, copy them to maintain the originals
	src = np.ones((m + 1, A.shape[0]))
	dst = np.ones((m + 1, B.shape[0]))
	src[:m, :] = np.copy(A.T)
	dst[:m, :] = np.copy(B.T)

	# apply the initial pose estimation
	if init_pose is not None:
		src = np.dot(init_pose, src)

	prev_error = 0

	for i in range(max_iterations):
		# find the nearest neighbors between the current source and destination points
		distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

		# compute the transformation between the current source and nearest destination points
		T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

		# update the current source
		src = np.dot(T, src)

		# check error
		mean_error = np.mean(distances)
		if np.abs(prev_error - mean_error) < tolerance:
			break
		prev_error = mean_error

	# calculate final transformation
	T, R, t = best_fit_transform(A, src[:m, :].T)

	return T, R, t, distances, i


if __name__ == '__main__':
	from mpl_toolkits.mplot3d import axes3d
	import matplotlib.pyplot as plt

	np.set_printoptions(suppress=True)

	# N = 10  # number of random points in the dataset
	# num_tests = 100  # number of test iterations
	# dim = 3  # number of dimensions of the points
	# noise_sigma = .01  # standard deviation error to be added
	# translation = .1  # max translation of the test set
	# rotation = .1  # max rotation (radians) of the test set
	#
	#
	# def rotation_matrix(axis, theta):
	# 	'''根据旋转轴和旋转角度，生成旋转矩阵'''
	# 	axis = axis / np.sqrt(np.dot(axis, axis))
	# 	a = np.cos(theta / 2.)
	# 	b, c, d = -axis * np.sin(theta / 2.)
	# 	return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
	# 	                 [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
	# 	                 [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
	#
	#
	# ##@@ 生成两组差了微小R,t的点云A,B.     B=R(A+t)
	# A = np.random.rand(N, dim)  # A:随机生成的一组点云数据
	# B = np.copy(A)  # B:在A点云基础上加入随机旋转和平移的第二组点云数据
	# t = np.random.rand(dim) * translation
	# B += t
	# R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
	# B = np.dot(R, B.T).T  # B=R(A+t)
	# B += np.random.randn(N, dim) * noise_sigma  # 加入噪声

	x = np.linspace(0,9,10)
	y = np.linspace(0,9,10)
	z = np.linspace(0,9,10)

	A = np.array(np.meshgrid(x,y,z))
	A = A.reshape(3,-1).T
	B = A+50.
	Rv = np.array([0.5,0.5,0.5])
	R,_ = cv2.Rodrigues(Rv)
	B = B.dot(R)
	print(B.shape)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='g')  # 绘制数据点
	ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='r', marker='*')  # 绘制数据点
	plt.show()

	##@@ icp算法的线性求解，该算法要求A,B点云完全配准对齐
	T, R1, t1 = best_fit_transform(A, B)

	##@@ ICP迭代法求解，该算法不要求A,B点云配准
	# 打乱A,B矩阵的对应顺序
	# temp = np.copy(B)
	# temp[0] = B[1]
	# temp[1] = B[0]
	# temp[2] = B[3]
	# temp[3] = B[2]
	# temp[4] = B[5]
	# temp[5] = B[4]
	# temp[6] = B[7]
	# temp[7] = B[6]
	# temp[8] = B[9]
	# temp[9] = B[8]
	# B = temp
	# T, R1, t1, distances, iterations = icp(A, B, tolerance=0.000001)

	##@@ 验证ICP求解的R,T能否实现点云数据的匹配
	C = (R1.dot(A.T) + t1).T
	print(B - C)


	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='r', marker='*')  # 绘制数据点
	ax.scatter(C[:, 0], C[:, 1], C[:, 2], c='g')  # 绘制数据点
	plt.show()
