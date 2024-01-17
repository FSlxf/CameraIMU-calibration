# -*- coding:utf-8 _*-
"""
@author:Chexqi
@time: 2019/03/26
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

axis_WCS = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1.]])


def ShowPoints3D(Points3D, c='r', xlim=None, ylim=None, zlim=None, title='ShowPoints3D'):
	'''显示N*3的3D点'''
	plt.figure(title)
	ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
	ax.scatter(Points3D[:, 0], Points3D[:, 1], Points3D[:, 2], c=c)  # 绘制数据点
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if zlim is not None:
		ax.set_zlim(zlim)
	# if xlim is None and ylim is None and zlim is None:
	# 	ax.set_aspect('equal')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')  # 坐标轴
	plt.show()


def ShowPoints2D(Points2D, c='r', xlim=None, ylim=None, title='ShowPoints2D'):
	'''显示N*2的2D点'''
	plt.figure(title)
	plt.scatter(Points2D[:, 0], Points2D[:, 1], c=c)  # 绘制数据点
	if xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		plt.ylim(ylim)
	if xlim is None and ylim is None:
		plt.axis('equal')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()


def pv_add_points(plotter, points3D, color='w', opacity=1):
	if points3D.shape[1]!=3:
		points3D = points3D.T
	actor_points = plotter.add_mesh(points3D, color=color, opacity=opacity)
	return actor_points


def pv_add_lines(plotter, points3D, color='w', opacity=1, line_width=5):
	'''通过3D点画线'''
	lines = pv.lines_from_points(points3D)
	actor_lines = plotter.add_mesh(lines, color=color, opacity=opacity, line_width=line_width)
	return actor_lines


def pv_add_arrow(plotter, start, direction, color='w', opacity=1):
	arrow = pv.Arrow(start=start, direction=direction, scale='auto')  # scale为auto的情况下，箭头长度和direction相关
	actor_arrow = plotter.add_mesh(arrow, color=color, opacity=opacity)
	return actor_arrow


def pv_add_axis_WCS(plotter, arrow_length, opacity=1, colors=['r', 'g', 'b']):
	'''添加世界坐标系'''
	coordinate_points = np.array([[0, 0, 0],  # 原点
	                              [arrow_length, 0, 0],  # x
	                              [0, arrow_length, 0],  # y
	                              [0, 0, arrow_length], ])  # z
	actor_arrow_x = pv_add_arrow(plotter, coordinate_points[0], coordinate_points[1], colors[0], opacity=opacity)
	actor_arrow_y = pv_add_arrow(plotter, coordinate_points[0], coordinate_points[2], colors[1], opacity=opacity)
	actor_arrow_z = pv_add_arrow(plotter, coordinate_points[0], coordinate_points[3], colors[2], opacity=opacity)
	return actor_arrow_x, actor_arrow_y, actor_arrow_z


def pv_add_axis_from4points(p, coordinate_points, simple_line=False, opacity=1, line_width=5, colors=['r', 'g', 'b']):
	'''
	:param p:
	:param coordinate_points:  4*3的点，第一行为原点，第2.3.4行依次为x.y.z轴，
	:param simple_line:  Ture的话用直线段表示坐标系，否则用箭头
	:return:
	'''
	if coordinate_points.shape[1] != 3:
		coordinate_points = coordinate_points.T
	if simple_line:
		actor_arrow_x = pv_add_lines(p, coordinate_points[[0, 1]], color=colors[0], opacity=opacity, line_width=line_width)
		actor_arrow_y = pv_add_lines(p, coordinate_points[[0, 2]], color=colors[1], opacity=opacity, line_width=line_width)
		actor_arrow_z = pv_add_lines(p, coordinate_points[[0, 3]], color=colors[2], opacity=opacity, line_width=line_width)
	else:
		actor_arrow_x = pv_add_arrow(p, coordinate_points[0], coordinate_points[1] - coordinate_points[0], colors[0], opacity=opacity)
		actor_arrow_y = pv_add_arrow(p, coordinate_points[0], coordinate_points[2] - coordinate_points[0], colors[1], opacity=opacity)
		actor_arrow_z = pv_add_arrow(p, coordinate_points[0], coordinate_points[3] - coordinate_points[0], colors[2], opacity=opacity)
	return actor_arrow_x, actor_arrow_y, actor_arrow_z


def pv_add_bounds_grid_pv(p, bounds, show_grid=True):
	'''
	通过插入几个坐标点 控制Pyvista显示的边缘
	:param bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
	:return:
	'''
	xmin, xmax, ymin, ymax, zmin, zmax = np.array(bounds).ravel()
	p.add_mesh(np.array([[xmin, ymin, zmin],
	                     [xmax, ymax, zmax]]), opacity=0)
	if show_grid:
		p.show_grid()
	else:
		p.show_bounds()


def ShowPoints3D_pv(Points3D, color='r', point_size=10, render_points_as_spheres=1, show_bounds=False, axis_length=None):
	'''显示N*3的3D点'''
	if Points3D.shape[1] != 3:
		Points3D = Points3D.T
	plotter = pv.Plotter()
	plotter.add_mesh(Points3D, color=color, point_size=point_size, render_points_as_spheres=render_points_as_spheres)
	if show_bounds:
		plotter.show_bounds()
	if axis_length is not None:
		pv_add_axis_WCS(plotter, axis_length)
	plotter.show()


# def ShowPoints3D_List_pv(Points3D_List, color=None, point_size=10, render_points_as_spheres=1, show_bounds=False, axis_length=None):
# 	'''显示多组N*3的3D点'''
# 	plotter = pv.Plotter()
# 	for Points3D in Points3D_List:
# 		if Points3D.shape[1] != 3:
# 			Points3D = Points3D.T
# 		if color is not None:
# 			plotter.add_mesh(Points3D, color=color, point_size=point_size, render_points_as_spheres=render_points_as_spheres)
# 		else:
# 			NewColor = np.random.rand(3)
# 			plotter.add_mesh(Points3D, color=NewColor, point_size=point_size, render_points_as_spheres=render_points_as_spheres)
# 	if show_bounds:
# 		plotter.show_bounds()
# 	if axis_length is not None:
# 		ShowAxis_pv(plotter, axis_length)
# 	plotter.show()


if __name__ == '__main__':
	Points3D = np.random.randint(0, 255, size=[40, 3])
	# ShowPoints3D(Points3D, xlim=(0, 300), ylim=(0, 150), zlim=(0, 300))
	ShowPoints3D_pv(Points3D, color='r', point_size=5)

# Points2D = np.random.randint(0, 255, size=[40, 2])
# ShowPoints2D(Points2D, xlim=(0, 300), ylim=(-30, 300))
