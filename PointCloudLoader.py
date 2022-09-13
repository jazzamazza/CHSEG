import numpy as np
import open3d as o3d
import laspy as lp
import sys
import importlib
import os

class PointCloudLoader:
  """Class responsible for loading point clouds"""
  def __init__(self, path="", downsample=False, ds_size=0, index=4):
    """Constructor
    args:
        path (file): file path
    """
    self.pcd_path = path     
    self.downsample = downsample
    self.ds_size = ds_size   
    self.final_pcd_file = 'finalPcd' + '_ds_' + str(downsample) + "_ds_size_" + str(ds_size) 
    self.final_pcd_all_file = 'finalPcdAll' + '_ds_' + str(downsample) + "_ds_size_" + str(ds_size)
    self.index = index

  def load_point_cloud(self, option='raw', load_downsampled=False, path1='', path2=''):
    '''Method to load point cloud from file, call helper methods, and create to numpy arrays: one containing the ground truth labels and one without
    args:
      option: the type of dataset to load from file (raw, cldCmp (dataset 2), or pnet (dataset 3)
      load_downsampled (bool): load a downsampled point cloud from file
      path1: path of downsampled finalpcd file
      path2: path of downsampled finalpcdAll file 
    returns:
      final_pcd: Point cloud without ground truth labels
      final_pcd_all: Point cloud with ground truth labels
    '''
    if load_downsampled:
          print("\n****************** Loading Downsampled Point Cloud *******************")
          final_pcd = np.load(path1)
          final_pcd_all = self.round_ground_truth(np.load(path2), option!='cldCmp')
    elif option!='cldCmp': 
      print("\n****************** Loading Point Cloud *******************")
      point_cloud = np.load(self.pcd_path)
      self.get_attributes(point_cloud)
      if option=='raw':
            final_pcd, final_pcd_all = self.load_raw_dataset(point_cloud)
      else:
            final_pcd, final_pcd_all = self.load_dataset_3(point_cloud)
    else:
      final_pcd, final_pcd_all = self.load_dataset_2()

    self.get_attributes(final_pcd, "final_pcd", self.final_pcd_file + '_' + option + '.npy') 
    self.get_attributes(final_pcd_all, "final_pcd_all", self.final_pcd_all_file + '_' + option + '.npy') 

    return final_pcd, final_pcd_all
        
  def load_raw_dataset(self, point_cloud):
    """Method to load raw point cloud, stored in a numpy file, perform downsampling, and create two point cloud datasets: one with and one without ground truth labels
    Args:
      point_cloud: the point cloud loaded into a npy array
    Returns:
        final_pcd: Point cloud without ground truth labels containing (x,y,z,intensity) features
        final_pcd_all: Point cloud with ground truth labels containing (x,y,z,intensity, truth_label) features
    """
    print("\n****************** Final Point Cloud *******************")  
    if self.downsample:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        rawFeatures = np.hstack((point_cloud[:,3:4], point_cloud[:,4:5], point_cloud[:,4:5])) # numpy array of (intensity, truth_label, truth_label)
        pcd.normals = o3d.utility.Vector3dVector(rawFeatures)
        final_pcd = pcd.voxel_down_sample(voxel_size=self.ds_size) # downsample pcd
        pc_points = np.asarray(final_pcd.points) # convert pcd points to np array
        pc_features = np.asarray(final_pcd.normals) # convert pcd additional features to np array
        final_pcd = np.delete(np.hstack((pc_points, pc_features)), [4,5], 1) # remove info unneccessary for clustering from pcd
        final_pcd_all = np.delete(np.hstack((pc_points, pc_features)), [5], 1)
        final_pcd_all[:,self.index:self.index+1] = np.ceil(final_pcd_all[:,self.index:self.index+1])
    else:
        final_pcd = point_cloud[:,:4] # without truth label
        final_pcd_all = point_cloud[:,:5]  

    return final_pcd, final_pcd_all

  def load_dataset_2(self):
    """Method to load CloudCompare generated point cloud, stored in a las file, perform downsampling, and create two point cloud datasets: one with and one without ground truth labels
    Returns:
        final_pcd: Point cloud without ground truth labels containing (x,y,z, CloudCompare generated features) features
        final_pcd_all: Point cloud with ground truth labels containing (x,y,z, truth_label, CloudCompare generated features) features
    """
    print("\n******************Loading Point Cloud with CloudCompare Generated Features*******************")
    with lp.open(self.pcd_path) as pcd_f:
        print("Header:", pcd_f.header, "\nPoints:", pcd_f.header.point_count)

    pcd = lp.read(self.pcd_path)
    print("Standard features:", list(pcd.point_format.standard_dimension_names))
    print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
    print("Extra feature count:", len(list(pcd.point_format.extra_dimension_names)))

    final_pcd = np.transpose(np.vstack((pcd.x, pcd.y, pcd.z)))

    for dim in pcd.point_format.extra_dimension_names:
        if dim != "truth":
          final_pcd = np.hstack((final_pcd, np.nan_to_num(np.vstack((pcd[dim])))))
        else:
          truth_label = np.nan_to_num(np.vstack((pcd[dim])))

    features = final_pcd[:, 3:]
    xyz_points = final_pcd[:, :3]
    labels_and_features = np.hstack((truth_label, features))
    final_pcd, final_pcd_all = self.get_point_clouds(xyz_points, features, labels_and_features, 18, True)

    return final_pcd, final_pcd_all

  def load_dataset_3(self, point_cloud):
    """Method to load PointNet++ generated point cloud, stored in a npy file, perform downsampling, and create two point cloud datasets: one with and one without ground truth labels
    Args:
      point_cloud: the point cloud loaded into a npy array
    Returns:
        final_pcd: Point cloud without ground truth labels containing (x,y,z, PointNet++ generated features) features
        final_pcd_all: Point cloud with ground truth labels containing (x,y,z, truth_label, PointNet++ generated features) features
    """
    # divide point_cloud into points, ground truth labels and features 
    points = point_cloud[:,:3]
    features = point_cloud[:,4:] 
    labels_and_features = point_cloud[:,3:]
    final_pcd, final_pcd_all = self.get_point_clouds(points, features, labels_and_features, 126, False)
    return final_pcd, final_pcd_all

  def get_point_clouds(self, points, features, labels_and_features, num_feats, div255):
    '''Method to create final point cloud datasets
    Args:
      points: a numpy array containing x,y,z values
      features: a numpy array containing features
      labels_and_features: a numpy array containing ground truth labels and features
      num_feats: the maximum number of features to include in the downsampled point cloud
      div255: whether the downsampled point cloud rounding needs to be divided by 255 or not
    Returns:
      final_pcd: Point cloud without ground truth labels
      final_pcd_all: Point cloud with ground truth labels
    '''
    if self.downsample:
        final_pcd = self.voxel_downsample(points, features, num_feats)
        final_pcd_all = self.voxel_downsample(points, labels_and_features, num_feats) 
        final_pcd_all = self.round_ground_truth(div255)
    else:
        final_pcd = np.hstack((points, features))
        final_pcd_all = np.hstack((points, labels_and_features))
    return final_pcd, final_pcd_all

  def convert_pcd(self, outputPath):
    '''Method to convert raw point cloud to a format readable by PointNet++ and downsample it by a given voxel. 
       Raw point cloud data is in the form: {x,y,z,intensity, ground truth}, but PointNet++
       expects data in the form: {x,y,z,r,g,b}, so we store the intensity values as {r,g,b} values
    Args:
      outputPath: the file path to save the converted point cloud
    Returns:
      outputPath: an updated file path where the converted point cloud is saved'''
    print("\n******************Convert Point Cloud to PointNet++ Readable Format*******************")

    inputPath = self.pcd_path
    pointCloud = np.load(inputPath)
    
    # divide point cloud into points, truthLabels and features 
    points = pointCloud[:,:3]
    intensity = pointCloud[:,3:4] 
    truthLabel = pointCloud[:,4:5] 
          
    # format using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
    features = np.hstack((intensity, intensity, intensity)) # form a 3D vector to add to o3d pcd
    pcd.colors = o3d.utility.Vector3dVector(features) # store intensity as every value in color vector
    labels = np.hstack((truthLabel, truthLabel, truthLabel)) # form a 3D vector to add to o3d pcd
    pcd.normals = o3d.utility.Vector3dVector(labels) # store intensity as every value in color vector
    print(pcd)
    downpcd = pcd.voxel_down_sample(voxel_size=self.ds_size)

    # save point cloud 
    outputPath = outputPath + "_ds_" + str(self.ds_size) + ".ply"
    o3d.io.write_point_cloud(outputPath, downpcd)

    return outputPath

  def load_pointnet_pcd(self):
    '''This method calls the PointNet++ code to add ML-generated features to the raw point cloud
    Args:
      pnet_input_file_path: file path of ply file containing point cloud to input to PointNet++
    Returns: 
      point cloud with PointNet++ generated features saved in a numpy array
    '''
    pnet_input_file_path = self.convert_pcd(self.pcd_path)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'PointNet++'))
    pnet = importlib.import_module('test_semseg')
    saved_pnet_file = 'pcd_with_pnet_features.npy'
    return pnet.main_semseg(pnet_input_file_path, saved_pnet_file)

  def voxel_downsample(self, points, features, upperBound):
        '''This method is responsible for performing voxel downsampling on a point cloud
        Args:
          points: a numpy array containing x,y,z values
          features: a numpy array containing features
          upperBound: the maximum number of features to include in the downsampled point cloud
        Returns:
          a numpy array containing a point cloud downsampled using a given voxel (self.ds_size)
        '''
        ds_points = np.array([])
        x,y = 0,0

        for _ in range(0, upperBound):  
            x += 1

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
            pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6])

            downpcd = pc.voxel_down_sample(voxel_size=self.ds_size)
            ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
            ds_points = np.asarray(downpcd.points)
            
            if x==1: total_ds_features = ds_features
            else: total_ds_features = np.hstack((total_ds_features, ds_features))   
            y = y + 6
            if y>=upperBound: break

        return np.hstack((ds_points, total_ds_features))

  def get_attributes(self, pcd, arr_name="Point Cloud", npy_name=None):
    """Prints attributes of given NumPy array to console
    Args:
        pcd (Any): Point Cloud in a NumPy array
        arr_name: string representing the name of the point cloud (default=Point Cloud)
        npy_name: name of .npy file to save the point cloud in
    """
    heading_label = arr_name+" Attributes:"
    heading_label += ('\n') + (len(heading_label)*'*')
    print("\n" + heading_label)
    print("\t- Point cloud size:", np.size(pcd))
    print("\t- Point cloud dim:", np.ndim(pcd))  
    print("\t- Point cloud shape:", np.shape(pcd))
    print("\t- Point cloud data type:", pcd.dtype,'\n')

    if npy_name: np.save(npy_name, pcd)
    
  def round_ground_truth(self, final_pcd_all, div255):
      '''Method to round the ground truth values that get altered during voxel downsampling, 
         ensuring that all ground truth values are either 0 or 1
      Args:
        final_pcd_all: numpy array containing point cloud with ground truth values
        div255: whether the rounded ground truth values should be divided by 255 or not'''
      if div255:
          final_pcd_all[:,self.index:self.index+1] = np.ceil(final_pcd_all[:,self.index:self.index+1]/255)
      else:
        final_pcd_all[:,self.index:self.index+1] = np.ceil(final_pcd_all[:,self.index:self.index+1])
      return final_pcd_all
