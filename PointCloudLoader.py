# PointCloudLoader
import numpy as np
import open3d as o3d
import laspy as lp

# Method to load and visualise a point cloud in a .npy file using open3d
def loadPointCloud_npy(vis):
     print("\n******************Loading Point Cloud with Raw Features (x, y, z, intensity) *******************")

     #load point cloud to numpy
     inputPath = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered.npy" #path to point cloud file
     pointCloud = np.load(inputPath)
     print("Point cloud size: ", pointCloud.size)
     
     # divide pointCloud into points and features 
     points = pointCloud[:,:3]
     intensity = pointCloud[:,3:4] 
     truthLabel = pointCloud[:,4:5] 
     finalPCD = np.hstack((points, intensity))
     print("finalPCD[0]:",finalPCD[0])
    
     if (vis):
         # format using open3d
         pcd = o3d.geometry.PointCloud()
         pcd.points = o3d.utility.Vector3dVector(pointCloud[:,:3]) # add {x,y,z} points to pcd
         zero = pointCloud[:,4:5] # placeholder
         rawFeatures = np.hstack((intensity, truthLabel, zero)) # form a 3D vector to add to o3d pcd
         pcd.normals = o3d.utility.Vector3dVector(rawFeatures) # store additional features (intensity & truth labels) in pcd.normals
         print(pcd)

         # visualise point cloud
         downpcd = pcd.voxel_down_sample(voxel_size=0.05) # downsample pcd
         o3d.visualization.draw_geometries([downpcd])
     
     #pc_points = np.asarray(downpcd.points) # convert pcd points to np array
     #pc_features = np.asarray(downpcd.normals) # convert pcd additional features to np array
     #pc = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
     #print("Downsampled Point cloud size: ", pc.size)
     #print("0 is:", pc[0])
     #finalPCD = np.delete(pc, [4,5], 1) # remove info unneccessary for clustering from pcd
     #print(finalPCD[0])
     
     return finalPCD

def loadPointCloud_las(vis):
      print("\n******************Loading Point Cloud with Cloud Compare Generated Features (x, y, z, intensity) *******************")
     
      path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered _cldCmp.las"
      pcd = lp.read(path)

      print("All features:", list(pcd.point_format.dimension_names))
      points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()            #lists x, y, z cooridnates 
      print("points", points)
      print("c-continguous:", points.flags['C_CONTIGUOUS'])
      
      cloudCompareFeatures = list(pcd.point_format.extra_dimension_names)
      print("Cloud Compare Features:", cloudCompareFeatures)
      planarity = np.vstack(pcd['Planarity (0.049006)'])
      intensity = np.vstack(pcd['NormalX'])
      anisotropy = np.vstack(pcd['Anisotropy (0.049006)'])
      linearity = np.vstack(pcd['Linearity (0.049006)'])
      surfaceVariation = np.vstack(pcd['Surface variation (0.049006)'])
      eigenentropy = np.vstack(pcd['Eigenentropy (0.049006)'])
      omnivariance = np.vstack(pcd['Omnivariance (0.049006)'])
      
      #########
      print("nan planarity?:",np.isnan(planarity).any())
      print("planarity size:", planarity.size)
      print("num nans in planarity:", np.count_nonzero(np.isnan(planarity)))
      print("planarity size:", planarity.size)
      planarity = np.nan_to_num(planarity)
      print("nan planarity?:",np.isnan(planarity).any())

      intensity = np.nan_to_num(intensity)
      anisotropy = np.nan_to_num(anisotropy)
      linearity = np.nan_to_num(linearity)
      surfaceVariation = np.nan_to_num(surfaceVariation)
      eigenentropy = np.nan_to_num(eigenentropy)
      ######

      features = np.hstack((planarity, anisotropy, linearity, surfaceVariation, eigenentropy, intensity))

      print("points:", points)
      print("nan points?:",np.isnan(points).any())
      print("features:", features)
      print("nan features?:",np.isnan(features).any())
      finalPCD = np.hstack((points, features))

      print("finalPCD 0:", finalPCD)
      print("nan finalPCD?:",np.isnan(finalPCD).any())

      # BELOW GIVES ARRAY NOT C-CONTIGUOUS ERROR 
      # add extra features to point cloud numpy array
      #count = 0
      #cloudCompareFeatures = list(pcd.point_format.extra_dimension_names)
      #for feature in cloudCompareFeatures:
      #  print("Feature:", feature)
      #  newFeature = np.vstack(pcd[feature])
      #  print(newFeature)
      #  newFeature = np.nan_to_num(newFeature)
      #  if (count==0): 
      #    finalPCD = np.hstack((points, newFeature))
      #  else:
      #    finalPCD = np.hstack((finalPCD, newFeature))
      #  count += 1
      #  #print("c-continguous:", finalPCD.flags['C_CONTIGUOUS'])
      #  print("finalPCD[0]:", finalPCD[0])
      #return finalPCD
      
      if (vis): 
        # format using open3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        cloudCompareFeatures_1 = np.hstack((planarity, anisotropy, linearity))
        cloudCompareFeatures_2 = np.hstack((surfaceVariation, eigenentropy, intensity)) # form a 3D vector to add to o3d pcd
        pc.normals = o3d.utility.Vector3dVector(cloudCompareFeatures_1) # store additional features (intensity & planarity) in pc.normals
        pc.colors = o3d.utility.Vector3dVector(cloudCompareFeatures_1) # store additional features (intensity & planarity) in pc.normals
        print(pc)

        # visualise point cloud
        downpcd = pc.voxel_down_sample(voxel_size=0.05) # downsample pc
        o3d.visualization.draw_geometries([downpcd])
     
        #pc_points = np.asarray(downpcd.points) # convert pc points to np array
        #pc_features = np.asarray(downpcd.normals) # convert pc additional features to np array
        #finalPCD = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
        #print("Downsampled Point cloud size: ", finalPCD.size)
        #print("0 is:", finalPCD[0])
     
      return finalPCD

# raw point cloud data = x, y, z, intensity
# but PointNet++ expects = x, y, z, r, g, b
# so we store intensity value as r, g, b
def convertPCD():
  print("\n******************Convert Point Cloud to PointNet++ Readable Format*******************")

  #load point cloud to numpy
  inputPath = "./data/church_registered.npy"  #path to point cloud file
  pointCloud = np.load(inputPath)
  print("Point cloud size: ", pointCloud.size)
     
  # divide pointCloud into points and features 
  points = pointCloud[:,:3]
  intensity = pointCloud[:,3:4] 
  truthLabel = pointCloud[:,4:5] 
    
  # format using open3d
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
  features = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
  pcd.colors = o3d.utility.Vector3dVector(features) # store intensity as every value in color vector
  print(pcd)

  # save point cloud 
  o3d.io.write_point_cloud("./data/church_registered_updated.ply", pcd)

# Method to load and visualise a point cloud in a .ply file using open3d
def loadPointCloud_ply():
     print("\n******************Loading Point Cloud (.ply) with Raw Features (x, y, z, intensity) *******************")

     #load point cloud .ply file
     path = "./data/church_registered_updated.ply"
     pcd = o3d.io.read_point_cloud(path)
     print(pcd)

     points = np.asarray(pcd.points)
     print("Points:\n", points)
     coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
     print("coord_min: ", coord_min)
     print("coord_max: ", coord_max)
     
     # visualise point cloud
     downpcd = pcd.voxel_down_sample(voxel_size=0.05)
     o3d.visualization.draw_geometries([downpcd], zoom=0.3412,
                                   front=[0.4257, -0.2125, -0.8795],
                                   lookat=[2.6172, 2.0475, 1.532],
                                   up=[-0.0694, -0.9768, 0.2024])
