# point_cloudLoader
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
  def __init__(self, path=""):
    """Constructor

    Args:
        path (file): file path
    """
    self.pcd_path = path        

  # Method to load and visualise a point cloud in a .npy file using open3d
  def load_point_cloud_npy(self, vis, downsample=False, ds_size=0):
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
    final_pcd_all = np.hstack((points, intensity, truth_label))
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
        downpcd = pcd.voxel_down_sample(voxel_size=ds_size) # downsample pcd
        pc_points = np.asarray(downpcd.points) # convert pcd points to np array
        pc_features = np.asarray(downpcd.normals) # convert pcd additional features to np array
        pc = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
        print("Downsampled Point cloud size: ", pc.size)
        down_finalPCD = np.delete(pc, [4,5], 1) # remove info unneccessary for clustering from pcd
        final_pcd_all = pc
        print("final_pcd_all[0]: ", final_pcd_all[0])
        final_pcd_all = np.delete(final_pcd_all, [5], 1)
        print("final_pcd_all[0]: ", final_pcd_all[0])
        print("ground truth in pc:", final_pcd_all[:,4:5])
        self.get_attributes(down_finalPCD, "final_pcd") 
        print(down_finalPCD[0])
        final_pcd = down_finalPCD
      
      
    np.save("FINAL-PCD_raw_0.085.npy", final_pcd)
    np.save("FINAL-PCD-ALL_raw_0.085.npy", final_pcd_all)

    return final_pcd, final_pcd_all
  
  def load_point_cloud_pNet_npy(self, vis, downsample=False, ds_size=0):
    """Method to load and visualise a point cloud stored as a .npy file
    Args:
        vis (bool): enable visualisation or now
    Returns:
        nparray: Point cloud as numpy array
    """
    print("\n** Loading Point Cloud **")
    point_cloud = np.load(self.pcd_path)
    self.get_attributes(point_cloud)

    # divide point_cloud into points and features 
    points = point_cloud[:,:3]
    labels = point_cloud[:,3:4]
    features = point_cloud[:,4:] #doesnt include labels
    labels_and_features = point_cloud[:,3:]

    print("points.size:", points.size, "features.size:", features.size)
    print("points.shape:", np.shape(points), "features.shape:", np.shape(features))
    print("labels:", labels)
    print("labels_and_features:", labels_and_features)

    print("\n** Final Point Cloud **")

    if (vis):
      pview = PointCloudViewer()
      pview.vis_npy(points)

    if downsample:
        final_pcd = self.voxel_downsample(points, features, 126, ds_size)
        final_pcd_all = self.voxel_downsample(points, labels_and_features, 126, ds_size) 
    else:
        final_pcd = np.hstack((points, features))
        final_pcd_all = np.hstack((points, labels, features))

    print("point size", points.size)
    print("final_pcd_all[0]",final_pcd_all[0] )

    self.get_attributes(final_pcd, "final_pcd") 
    self.get_attributes(final_pcd_all, "final_pcd_all")
    print("hstacked pcd[0]:",final_pcd[0])

    np.save("FINAL-PCD_pointnet_0.05.npy", final_pcd)
    np.save("FINAL-PCD-ALL_PCD_pointnet_0.05.npy", final_pcd_all)

    return final_pcd, final_pcd_all

  def load_point_cloud_las(self, vis, downsample=False, ds_size=0):
        print("\n******************Loading Point Cloud with Cloud Compare Generated Features (x, y, z, intensity) *******************")

        path = self.pcd_path

        # understand las header data
        with lp.open(path) as pcd_f:
            print("Header:", pcd_f.header)
            point_count = pcd_f.header.point_count
            print("Points:", point_count)

        print("***READING LAS****")
        pcd = lp.read(path)
        print("Std features:", list(pcd.point_format.standard_dimension_names))
        print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
        geofeat_count = len(list(pcd.point_format.extra_dimension_names))
        print("Extra feat count:", geofeat_count)

        points = np.transpose(np.vstack((pcd.x, pcd.y, pcd.z)))
        print("points", points)

        for dim in pcd.point_format.extra_dimension_names:
            if dim != "truth":
              points = np.hstack((points, np.nan_to_num(np.vstack((pcd[dim])))))
            else:
                  truth_label = np.nan_to_num(np.vstack((pcd[dim])))
        final_features = points[:, 3:]
        xyz_points = points[:, :3]
        label_and_final_features = np.hstack((truth_label, final_features))

        print("TRUTH LABEL:", truth_label)
        print("label_and_final_features[0]:", label_and_final_features[0])

        self.get_attributes(points, "points pcd")
        print("points pcd", points)
        self.get_attributes(final_features, "final extra array")
        print("final extras", final_features)

        if vis:
            print("vis")

        if downsample:
            final_pcd = self.voxel_downsample(xyz_points, final_features, 12, ds_size)
            final_pcd_all = self.voxel_downsample(xyz_points, label_and_final_features, 12, ds_size)
        else:
            final_pcd = points
            final_pcd_all = np.hstack((truth_label, points))
            print("final_pcd_all[0]", final_pcd_all[0])
            print("final_pcd_all[:,3]", final_pcd[:,3])
        
        # np.save("FINAL-PCD_cloudCompare_0.085.npy", final_pcd)
        # np.save("FINAL-PCD-ALL_PCD_cloudCompare_0.085.npy", final_pcd_all)

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

  def voxel_downsample(self, points, features, upperBound, ds_size):
        ds_points = np.array([])
        x,y = 0,0

        for _ in range(0, upperBound):  
            x += 1
            print("===========================i:", y)
            print("points.size:", points.size, "points.shape:", np.shape(points))
            print("features.size:", features.size, "features.shape:", np.shape(features))

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
            pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6])

            downpcd = pc.voxel_down_sample(voxel_size=ds_size)
            ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
            ds_points = np.asarray(downpcd.points)
            print("ds_points.size:", ds_points.size, "ds_points shape:", np.shape(ds_points))
            print("ds_features.size:", ds_features.size, "ds_features shape:", np.shape(ds_features))
            
            if x==1: total_ds_features = ds_features
            else: total_ds_features = np.hstack((total_ds_features, ds_features))   
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
