from PointCloudLoader import PointCloudLoader
import numpy as np
import open3d as o3d
from Clustering import Clustering

def voxel_downsample_cldCmp():
    file_path = "/content/drive/Shareddrives/Thesis/Data/church_registered_cloudCompare.las"
    pc_loader = PointCloudLoader(file_path)
    points, features = pc_loader.load_point_cloud_las(False)
    print("Features shape:", np.shape(features))

    colours, normals, ds_points = np.array([]), np.array([]), np.array([])
    x,y = 0,0

    for i in range(0, 12):  
        x += 1
        print("i:", y)
        print("points.size:", points.size, "colours.size:", colours.size, "normals.size:", normals.size)
        print("points.shape:", np.shape(points), "colours.shape:", np.shape(colours), "normals.shape:", np.shape(normals))
        # print("colors", colours)
        # print("normals", normals)
        # print("points:", points)

        #o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
        if (y+3<=12): pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6]) 
        
        downpcd = pc.voxel_down_sample(voxel_size=0.5)
        if (y+3<=12): ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
        else: ds_features = np.asarray(downpcd.normals)
        # print("ds_features:", ds_features)
        print("ds_features.size:", ds_features.size, "ds_features shape:", np.shape(ds_features))
        ds_points = np.asarray(downpcd.points)
        # print("ds_points:", ds_points)
        print("ds_points.size:", ds_points.size, "ds_points shape:", np.shape(ds_points))
        if x==1: 
          old_ds_points = ds_points
          total_ds_features = ds_features
        else:
          print("EQUAL????",np.array_equal(old_ds_points, ds_points))
          old_ds_points = ds_points
          # print("total_ds_features:", total_ds_features)
          print("total_ds_features.size:", total_ds_features.size, "total_ds_features shape:", np.shape(total_ds_features))
          total_ds_features = np.hstack((total_ds_features, ds_features))

        colours, normals = np.array([]), np.array([])
        y = y + 6
        if y>=15: break

    # print("final downsampled points:", ds_points)
    # print("final downsampledfeatures:", total_ds_features)
    finalPCD = np.hstack((ds_points, total_ds_features))
    print("finalPCD.size:", finalPCD.size, "finalPCD.shape():", np.shape(finalPCD))
    print("finalPCD[0]:", finalPCD[0])

    clustering = Clustering(finalPCD, "2")
    clustering.k_means_clustering_faiss(13, "")

def voxel_downsample_pNet():
    file_path = "/content/drive/Shareddrives/Thesis/Data/church_registered_cloudCompare.las"
    pc_loader = PointCloudLoader(file_path)
    points, features = pc_loader.load_point_cloud_las(False)
    print("Features shape:", np.shape(features))

    colours, normals, ds_points = np.array([]), np.array([]), np.array([])
    x,y = 0,0

    for i in range(0, 12):  
        x += 1
        print("i:", y)
        print("points.size:", points.size, "colours.size:", colours.size, "normals.size:", normals.size)
        print("points.shape:", np.shape(points), "colours.shape:", np.shape(colours), "normals.shape:", np.shape(normals))
        # print("colors", colours)
        # print("normals", normals)
        # print("points:", points)

        #o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
        if (y+3<=12): pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6]) 
        
        downpcd = pc.voxel_down_sample(voxel_size=0.5)
        if (y+3<=12): ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
        else: ds_features = np.asarray(downpcd.normals)
        # print("ds_features:", ds_features)
        print("ds_features.size:", ds_features.size, "ds_features shape:", np.shape(ds_features))
        ds_points = np.asarray(downpcd.points)
        # print("ds_points:", ds_points)
        print("ds_points.size:", ds_points.size, "ds_points shape:", np.shape(ds_points))
        if x==1: 
          old_ds_points = ds_points
          total_ds_features = ds_features
        else:
          print("EQUAL????",np.array_equal(old_ds_points, ds_points))
          old_ds_points = ds_points
          # print("total_ds_features:", total_ds_features)
          print("total_ds_features.size:", total_ds_features.size, "total_ds_features shape:", np.shape(total_ds_features))
          total_ds_features = np.hstack((total_ds_features, ds_features))

        colours, normals = np.array([]), np.array([])
        y = y + 6
        if y>=15: break

    # print("final downsampled points:", ds_points)
    # print("final downsampledfeatures:", total_ds_features)
    finalPCD = np.hstack((ds_points, total_ds_features))
    print("finalPCD.size:", finalPCD.size, "finalPCD.shape():", np.shape(finalPCD))
    print("finalPCD[0]:", finalPCD[0])

    clustering = Clustering(finalPCD, "2")
    clustering.k_means_clustering_faiss(13, "")

def voxel_downsample_pNet():
    file_path = "/content/drive/MyDrive/PNET/Data/church_registered_pnet_final.npy"
    pc_loader = PointCloudLoader(file_path)
    points, features = pc_loader.load_point_cloud_npyPNET(False)
    print("Features shape:", np.shape(features))

    colours, normals, ds_points = np.array([]), np.array([]), np.array([])
    x,y = 0,0

    for i in range(0, 126):  
        x += 1
        print("===========================i:", y)
        print("points.size:", points.size)
        print("points.shape:", np.shape(points))
        

        #o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.normals = o3d.utility.Vector3dVector(features[:,y:y+3])
        pc.colors = o3d.utility.Vector3dVector(features[:,y+3:y+6])

        print("colors", pc.colors)
        print("normals", pc.normals)
        # print("points:", points)

        downpcd = pc.voxel_down_sample(voxel_size=0.5)
        ds_features = np.hstack((np.asarray(downpcd.normals), np.asarray(downpcd.colors)))
        
        # print("ds_features:", ds_features)
        print("ds_features.size:", ds_features.size, "ds_features shape:", np.shape(ds_features))
        ds_points = np.asarray(downpcd.points)
        # print("ds_points:", ds_points)
        print("ds_points.size:", ds_points.size, "ds_points shape:", np.shape(ds_points))
        if x==1: 
          old_ds_points = ds_points
          total_ds_features = ds_features
        else:
          print("EQUAL????",np.array_equal(old_ds_points, ds_points))
          old_ds_points = ds_points
          # print("total_ds_features:", total_ds_features)
          print("total_ds_features.size:", total_ds_features.size, "total_ds_features shape:", np.shape(total_ds_features))
          total_ds_features = np.hstack((total_ds_features, ds_features))

        colours, normals = np.array([]), np.array([])
        y = y + 6
        if y>=126: break

    # print("final downsampled points:", ds_points)
    # print("final downsampledfeatures:", total_ds_features)
    finalPCD = np.hstack((ds_points, total_ds_features))
    print("finalPCD.size:", finalPCD.size, "finalPCD.shape():", np.shape(finalPCD))
    print("finalPCD[0]:", finalPCD[0])

    clustering = Clustering(finalPCD, "2")
    clustering.k_means_clustering_faiss(13, "")

if __name__ == "__main__":
  voxel_downsample_cldCmp()
  voxel_downsample_pNet()
 
  
 
    