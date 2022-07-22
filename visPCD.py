import open3d as o3d
import numpy as np
import PointCloudLoader
import pptk

# print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud("./Data/church_registered_downsampled_0.05.ply")
# o3d.visualization.draw_geometries([pcd])

# pcd = o3d.io.read_point_cloud("./Data/church_registered _cloudCompare.las")
# o3d.visualization.draw_geometries([pcd])

# pcd = o3d.io.read_point_cloud("./Data/church_registered_downsampled_0.05.ply")
# o3d.visualization.draw_geometries([pcd],0.5)

print('pptk test')

# x = np.random.rand(100, 3)
# v = pptk.viewer(x)
# v.set(point_size=0.01)

xyz = pptk.rand(100, 3)
rgb = pptk.rand(100, 3)
v = pptk.viewer(xyz, rgb)
v.set(point_size=0.005)