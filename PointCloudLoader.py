# point_cloudLoader
from curses import has_colors
import numpy as np
import open3d as o3d
import laspy as lp
import sys
import importlib
import os
from PointCloudViewer import PointCloudViewer
 

class PointCloudLoader:
  """Point cloud loader
  """
  def __init__(self, path):
    """Constructor

    Args:
        path (file): file path
    """
    self.pcd_path = path

  # Method to load and visualise a point cloud in a .npy file using open3d
  def load_point_cloud_npy(self, vis, downsample=False):
    """Method to load and visualise a point cloud stored as a .npy file

    Args:
        vis (bool): enable visualisation or now

    Returns:
        nparray: Point cloud as numpy array
    """
    print("\n****************** Loading Point Cloud *******************")
    point_cloud = np.load(self.pcd_path)
    self.get_attributes(point_cloud)   
    # divide point_cloud into points and features 
    print("original pcd[0]:",point_cloud[0])
    points = point_cloud[:,:3]
    print("points[0]",points[0])
    intensity = point_cloud[:,3:4]
    print("intensity[0]",intensity[0])
    truth_label = point_cloud[:,4:5]
    print("truth label[0]",truth_label[0]) 
    
    print("\n****************** Final Point Cloud *******************")
    final_pcd = np.hstack((points, intensity)) #without truth label
    self.get_attributes(final_pcd, "final_pcd") 
    print("hstacked pcd[0]:",final_pcd[0])
    
    if (vis):
      pview = PointCloudViewer()
      pview.vis_npy(points, intensity, truth_label)
      
    if downsample:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        rawFeatures = np.hstack((intensity, truth_label, truth_label))
        pcd.normals = o3d.utility.Vector3dVector(rawFeatures)
        downpcd = pcd.voxel_down_sample(voxel_size=2) # downsample pcd
        pc_points = np.asarray(downpcd.points) # convert pcd points to np array
        pc_features = np.asarray(downpcd.normals) # convert pcd additional features to np array
        pc = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
        print("Downsampled Point cloud size: ", pc.size)
        down_finalPCD = np.delete(pc, [4,5], 1) # remove info unneccessary for clustering from pcd
        self.get_attributes(down_finalPCD, "final_pcd") 
        print(down_finalPCD[0])
        final_pcd = down_finalPCD

    return final_pcd
  
  def load_point_cloud_pNet_npy(self, vis, downsample=False):
    """Method to load and visualise a point cloud stored as a .npy file

    Args:
        vis (bool): enable visualisation or now

    Returns:
        nparray: Point cloud as numpy array
    """
    print("\n****************** Loading Point Cloud *******************")
    point_cloud = np.load(self.pcd_path)
    self.get_attributes(point_cloud)   

    # divide point_cloud into points and features 
    points = point_cloud[:,:3]
    features = point_cloud[:,3:]

    print("points.size:", points.size, "features.size:", features.size)
    print("points.shape:", np.shape(points), "features.shape:", np.shape(features))    
    
    print("\n****************** Final Point Cloud *******************")
    
    if (vis):
      pview = PointCloudViewer()
      pview.vis_npy(points)
    
    if downsample:
        final_pcd = self.voxel_downsample(points, features, upperBound=126)
    else:
        final_pcd = np.hstack((points, features))
    self.get_attributes(final_pcd, "final_pcd") 
    print("hstacked pcd[0]:",final_pcd[0])
    return final_pcd

  def load_point_cloud_las(self, vis, downsample=False):
      print("\n******************Loading Point Cloud with Cloud Compare Generated Features (x, y, z, intensity) *******************")
    
      path = self.pcd_path
      
      #understand las header data
      with lp.open(path) as pcd_f:
        print(pcd_f.header)
      
      pcd = lp.read(path)

      print("All features:", list(pcd.point_format.dimension_names))
      points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose() #lists x, y, z cooridnates 
      print("points", points)
      
      cloudCompareFeatures = list(pcd.point_format.extra_dimension_names)
    
      print("Cloud Compare Features:", cloudCompareFeatures)
      planarity = np.vstack(pcd['Planarity (0.049006)'])
      intensity = np.vstack(pcd['NormalX'])
      anisotropy = np.vstack(pcd['Anisotropy (0.049006)'])
      linearity = np.vstack(pcd['Linearity (0.049006)'])
      surfaceVariation = np.vstack(pcd['Surface variation (0.049006)'])
      eigenentropy = np.vstack(pcd['Eigenentropy (0.049006)'])
      omnivariance = np.vstack(pcd['Omnivariance (0.049006)'])
      eigenvalues_sum = np.vstack(pcd['Eigenvalues sum (0.049006)'])
      pca1 = np.vstack(pcd['PCA1 (0.049006)'])
      pca2 = np.vstack(pcd['PCA2 (0.049006)'])
      sphericity = np.vstack(pcd['Sphericity (0.049006)'])
      verticality = np.vstack(pcd['Verticality (0.049006)'])
      first_eigen = np.vstack(pcd['1st eigenvalue (0.049006)'])
      second_eigen = np.vstack(pcd['2nd eigenvalue (0.049006)'])
      third_eigen = np.vstack(pcd['3rd eigenvalue (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
    #   new_feat = np.vstack(pcd['feat (0.049006)'])
      
      #########
      planarity = np.nan_to_num(planarity)
      intensity = np.nan_to_num(intensity)
      anisotropy = np.nan_to_num(anisotropy)
      linearity = np.nan_to_num(linearity)
      surfaceVariation = np.nan_to_num(surfaceVariation)
      eigenentropy = np.nan_to_num(eigenentropy)
      omnivariance = np.nan_to_num(omnivariance)
      eigenvalues_sum= np.nan_to_num(eigenvalues_sum)
      pca1 = np.nan_to_num(pca1)
      pca2 = np.nan_to_num(pca2)
      sphericity = np.nan_to_num(sphericity)
      verticality = np.nan_to_num(verticality)
      first_eigen = np.nan_to_num(first_eigen)
      second_eigen = np.nan_to_num(second_eigen)
      third_eigen = np.nan_to_num(third_eigen)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
    #   new_feat = np.nan_to_num(new_feat)
      ######

      features = np.hstack((planarity, anisotropy, linearity, surfaceVariation, eigenentropy, intensity))
      features1 = np.hstack((omnivariance, eigenvalues_sum, pca1, pca2, sphericity, verticality))
      features2 = np.hstack((first_eigen, second_eigen, third_eigen))
    #   features3 = np.hstack((new_feat, new_feat, new_feat, new_feat, new_feat, new_feat, new_feat, new_feat))
   
      final_features = np.hstack((features, features1, features2))
    #   final_features = np.hstack((features, features1, features2, features3))
      

      if downsample:
            final_pcd = self.voxel_downsample(points, final_features, upperBound=18)
      else:
            final_pcd = np.hstack((points, final_features))

      if (vis): 
        # format using open3d
        pc = o3d.geometry.point_cloud()
        pc.points = o3d.utility.Vector3dVector(points)
        cloudCompareFeatures_1 = np.hstack((planarity, anisotropy, linearity))
        cloudCompareFeatures_2 = np.hstack((surfaceVariation, eigenentropy, intensity)) # form a 3D vector to add to o3d pcd
        pc.normals = o3d.utility.Vector3dVector(cloudCompareFeatures_1) # store additional features (intensity & planarity) in pc.normals
        pc.colors = o3d.utility.Vector3dVector(cloudCompareFeatures_2) # store additional features (intensity & planarity) in pc.normals
        print(pc)

        # visualise point cloud
        downpcd = pc.voxel_down_sample(voxel_size=0.05) # downsample pc
        o3d.visualization.draw_geometries([downpcd])
    
      return final_pcd

  def convert_pcd(self):
    
    # raw point cloud data = x, y, z, intensity
    # but PointNet++ expects = x, y, z, r, g, b
    # so we store intensity value as r, g, b
    print("\n******************Convert Point Cloud to PointNet++ Readable Format*******************")

    #load point cloud to numpy
    path = self.pcd_path  #path to point cloud file
    point_cloud = np.load(path)
    print("Point cloud size: ", point_cloud.size)
    
    # divide point_cloud into points and features 
    points = point_cloud[:,:3]
    intensity = point_cloud[:,3:4] 
    truthLabel = point_cloud[:,4:5] 
      
    # format using open3d
    pcd = o3d.geometry.point_cloud()
    pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
    features = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
    pcd.colors = o3d.utility.Vector3dVector(features) # store intensity as every value in color vector
    print(pcd)

    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # save point cloud 
    o3d.io.write_point_cloud("./Data/church_registered_updated.ply", downpcd)

  # Method to load and visualise a point cloud in a .ply file using open3d
  def load_point_cloud_ply(self, vis):
    print("\n******************Loading Point Cloud (.ply) with Raw Features (x, y, z, intensity) *******************")

    #load point cloud .ply file
    path = self.pcd_path
    pcd = o3d.io.read_point_cloud(path, print_progress=True)
    print("Point Cloud Loaded:", pcd)
    
    has_points = pcd.has_points()
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()
    has_covariances = pcd.has_covariances()
    print("pcd has points ->", has_points)
    if has_points:
      print(np.asarray(pcd.points))
    print("pcd has colours ->", has_colors)
    if has_colors:
      print(np.asarray(pcd.colors))
    print("pcd has normals ->", has_normals)
    if has_normals:
      print(np.asarray(pcd.normals))
    print("pcd has covariances ->", has_covariances)
    if has_covariances:
      print(np.asarray(pcd.covariances))
      
    pcd_points = np.asarray(pcd.points)
    pcd_npy = np.copy(pcd_points)
    
    if (vis):
      pview = PointCloudViewer()
      pview.vis_ply(pcd)
      
    return pcd_npy

  def loadPointCloud_pNet(self, vis):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'PointNet++'))
    pnet = importlib.import_module('test_semseg')
    return pnet.main_semseg()

  def voxel_downsample(points, features, upperBound):
        print("Features shape:", np.shape(features))

        ds_points = np.array([])
        x,y = 0,0

        for i in range(0, upperBound):  
            x += 1
            print("===========================i:", y)
            print("points.size:", points.size, "points.shape:", np.shape(points))

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
            pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6])

            downpcd = pc.voxel_down_sample(voxel_size=0.5)
            ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
            ds_points = np.asarray(downpcd.points)
            print("ds_features.size:", ds_features.size, "ds_features shape:", np.shape(ds_features))
            print("ds_points.size:", ds_points.size, "ds_points shape:", np.shape(ds_points))
            
            if x==1: 
                old_ds_points = ds_points
                total_ds_features = ds_features
            else:
                print("ds_point and old_ds_points EQUAL?",np.array_equal(old_ds_points, ds_points))
                old_ds_points = ds_points
                total_ds_features = np.hstack((total_ds_features, ds_features))
                print("total_ds_features.size:", total_ds_features.size, "total_ds_features shape:", np.shape(total_ds_features))
                
            y = y + 6
            if y>=upperBound: break

        finalPCD = np.hstack((ds_points, total_ds_features))
        print("finalPCD.size:", finalPCD.size, "finalPCD.shape():", np.shape(finalPCD))
        print("finalPCD[0]:", finalPCD[0])

        return finalPCD

  def get_attributes(self, pcd, arr_name="Point Cloud"):
    """Prints attributes of given numpy array to console

    Args:
        pcd (Any): Point Cloud Array
    """
    heading_label = arr_name+" Attributes:"
    heading_label += ('\n') + (len(heading_label)*'*')
    print("\n" + heading_label)
    print("\t- Point cloud size:", np.size(pcd))
    print("\t- Point cloud dim:", np.ndim(pcd))  
    print("\t- Point cloud shape:", np.shape(pcd))
    print("\t- Point cloud data type:", pcd.dtype,'\n')

