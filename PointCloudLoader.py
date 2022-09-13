import numpy as np
import open3d as o3d
import laspy as lp
import sys
import importlib
import os

class PointCloudLoader:
  """Class responsible for loading point clouds"""
  def __init__(self, path=""):
    """Constructor
    args:
        path (file): file path
    """
    self.pcd_path = path        

  def load_point_cloud_npy(self, downsample=False, ds_size=0):
    """Method to load and visualise a point cloud stored as a .npy file
    Args:
        vis (bool): enable visualisation or not
    Returns:
        nparray: Point cloud as numpy array
    """
    print("\n****************** Loading Point Cloud *******************")
    point_cloud = np.load(self.pcd_path)
    self.get_attributes(point_cloud)   
    
    # divide point_cloud into points and features 
    intensity = point_cloud[:,3:4]
    truth_label = point_cloud[:,4:5]
    
    print("\n****************** Final Point Cloud *******************")
    final_pcd = point_cloud[:,:4] #without truth label
    final_pcd_all = point_cloud[:,:5]
    self.get_attributes(final_pcd, "final_pcd") 
      
    if downsample:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        rawFeatures = np.hstack((intensity, truth_label, truth_label))
        pcd.normals = o3d.utility.Vector3dVector(rawFeatures)
        final_pcd = pcd.voxel_down_sample(voxel_size=ds_size) # downsample pcd
        final_pcd = np.delete(final_pcd_all, [4,5], 1) # remove info unneccessary for clustering from pcd
        
        pc_points = np.asarray(final_pcd.points) # convert pcd points to np array
        pc_features = np.asarray(final_pcd.normals) # convert pcd additional features to np array
        final_pcd_all = np.hstack((pc_points, pc_features[:2])) # concatenate the 2 np arrays
        # final_pcd_all = np.delete(final_pcd_all, [5], 1)

    self.get_attributes(final_pcd, "final_pcd", "FINAL-PCD_raw_0.085.npy") 
    self.get_attributes(final_pcd_all, "final_pcd_all", "FINAL-PCD-ALL_raw_0.085.npy") 

    return final_pcd, final_pcd_all
  
  def load_point_cloud_pNet_npy(self, downsample=False, ds_size=0):
    """Method to load and visualise a point cloud stored as a .npy file
    Args:
        vis (bool): enable visualisation or now
    Returns:
        nparray: Point cloud as numpy array
    """
    point_cloud = np.load(self.pcd_path)
    self.get_attributes(point_cloud)

    # divide point_cloud into points and features 
    points = point_cloud[:,:3]
    features = point_cloud[:,4:] 
    labels_and_features = point_cloud[:,3:]

    final_pcd, final_pcd_all = self.get_point_clouds(downsample, points, features, labels_and_features, ds_size, 126)

    self.get_attributes(final_pcd, "final_pcd", "FINAL-PCD_pointnet_0.05.npy", "FINAL-PCD-ALL_PCD_pointnet_0.05.npy") 
    self.get_attributes(final_pcd_all, "final_pcd_all")

    return final_pcd, final_pcd_all

  def get_point_clouds(self, downsample, points, features, labels_and_features, ds_size, num_feats):
    if downsample:
        final_pcd = self.voxel_downsample(points, features, num_feats, ds_size)
        final_pcd_all = self.voxel_downsample(points, labels_and_features, num_feats, ds_size) 
    else:
        final_pcd = np.hstack((points, features))
        final_pcd_all = np.hstack((points, labels_and_features))
    return final_pcd, final_pcd_all

  def load_point_cloud_las(self, downsample=False, ds_size=0):
        print("\n******************Loading Point Cloud with Cloud Compare Generated Features (x, y, z, intensity) *******************")
        path = self.pcd_path

        with lp.open(path) as pcd_f:
            print("Header:", pcd_f.header)
            print("Points:", pcd_f.header.point_count)

        print("***READING LAS****")
        pcd = lp.read(path)
        print("Std features:", list(pcd.point_format.standard_dimension_names))
        print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
        print("Extra feat count:", len(list(pcd.point_format.extra_dimension_names)))

        final_pcd = np.transpose(np.vstack((pcd.x, pcd.y, pcd.z)))

        for dim in pcd.point_format.extra_dimension_names:
            if dim != "truth":
              final_pcd = np.hstack((final_pcd, np.nan_to_num(np.vstack((pcd[dim])))))
            else:
              truth_label = np.nan_to_num(np.vstack((pcd[dim])))

        features = final_pcd[:, 3:]
        xyz_points = final_pcd[:, :3]
        labels_and_features = np.hstack((truth_label, features))
        final_pcd, final_pcd_all = self.get_point_clouds(downsample, xyz_points, features, labels_and_features, ds_size, 18)
        
        self.get_attributes(final_pcd, "final_pcd", "FINAL-PCD_cloudCompare_0.085.npy") 
        self.get_attributes(final_pcd_all, "final_pcd_all", "FINAL-PCD-ALL_PCD_cloudCompare_0.085.npy")

        return final_pcd, final_pcd_all

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

  def loadPointCloud_pNet(self):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'PointNet++'))
    pnet = importlib.import_module('test_semseg')
    return pnet.main_semseg()

  def voxel_downsample(self, points, features, upperBound, ds_size):
        ds_points = np.array([])
        x,y = 0,0

        for _ in range(0, upperBound):  
            x += 1

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
            pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6])

            downpcd = pc.voxel_down_sample(voxel_size=ds_size)
            ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
            ds_points = np.asarray(downpcd.points)
            
            if x==1: total_ds_features = ds_features
            else: total_ds_features = np.hstack((total_ds_features, ds_features))   
            y = y + 6
            if y>=upperBound: break

        finalPCD = np.hstack((ds_points, total_ds_features))

        return finalPCD

  def get_attributes(self, pcd, arr_name="Point Cloud", npy_name=None):
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

    if npy_name: np.save(npy_name, pcd)
    
  def load_ds_from_file(self, path1, path2, index, div255=False):
      '''Load Downsampled Point Cloud from File
      args:
        path1: path of point cloud without ground truth labels
        path2: path of point cloud with ground truth labels
        index: index of ground truth in point cloud with ground truth labels
        div255 (bool): whether the ground truth labels should be divided by 255 or not'''
      final_pcd = np.load(path1)
      self.get_attributes(final_pcd)   
      final_pcd_all = np.load(path2)
      if div255:
        final_pcd_all[:,index:index+1] = np.ceil(final_pcd_all[:,index:index+1]/255)
      else:
        final_pcd_all[:,index:index+1] = np.ceil(final_pcd_all[:,index:index+1])
      self.get_attributes(final_pcd_all)
      return final_pcd, final_pcd_all
