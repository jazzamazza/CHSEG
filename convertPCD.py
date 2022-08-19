import numpy as np
import open3d as o3d

# CHANGE PATHS:
rootPath = "Data/"

inputPath = rootPath+"church_registered.npy"  #path to point cloud file

pointCloud = np.load(inputPath)
print("Point cloud size: ", pointCloud.size)
    
# divide pointCloud into points and features 
points = pointCloud[:,:3]
intensity = pointCloud[:,3:4] 
truthLabel = pointCloud[:,4:5] 

for i in truthLabel:
  if i[0] != float(1) and i[0] != float(0):
      print("oh no", i[0])

# format using open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
features = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
pcd.colors = o3d.utility.Vector3dVector(features) # store intensity as every value in color vector
labels = np.hstack((truthLabel, truthLabel, truthLabel)) # form a 3D vector to add to o3d pcd
pcd.normals = o3d.utility.Vector3dVector(labels) # store intensity as every value in color vector
print(pcd)

downsample_size = 0.075

downpcd = pcd.voxel_down_sample(voxel_size=downsample_size)

outputPath = rootPath+"church_registered_downsampled_"+str(downsample_size)+".ply"

# save point cloud 
o3d.io.write_point_cloud(outputPath, downpcd)