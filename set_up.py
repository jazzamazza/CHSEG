import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud):
          self.pcd = pointCloud
          
     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=k) # number of clusters (k)
          kmeans.fit(self.pcd) # apply k means to dataset
          
          print("\nCluster centres:")
          print(kmeans.cluster_centers_)
          print("\nLabels:")
          print(kmeans.labels_)

# Tester method to test Clustering class and k_means algorithm          
def testMethod():
     X = np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])
     
     clustering = Clustering(X)
     clustering.k_means_clustering(4)

# Method to load and visualise a point cloud in a .npy file using open3d
def loadPointCloud_npy():
     print("\nLOAD NPY POINT CLOUD DATA")

     #load point cloud to numpy
     inputPath = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered.npy" #path to point cloud file
     pointCloud = np.load(inputPath)
     print("Point cloud size: ", pointCloud.size)
     
     # format using open3d
     pcd = o3d.geometry.PointCloud()
     pcd.points = o3d.utility.Vector3dVector(pointCloud[:,:3])
     print(pcd)
     
     print("1: ", pcd.has_colors())

     # visualise point cloud
     # my laptop cannot render the normal point cloud, so I had to downsample it 
     downpcd = pcd.voxel_down_sample(voxel_size=0.05)
     o3d.visualization.draw_geometries([downpcd])
     
     return pointCloud

# Method to load and visualise a point cloud in a .ply file using open3d
def loadPointCloud_ply():
     print("\nLOAD PLY POINT CLOUD DATA")

     #load point cloud .ply file
     path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered.ply"
     pcd = o3d.io.read_point_cloud(path)
     print(pcd)
     print(np.asarray(pcd.points))

     # my laptop cannot render the normal point cloud, so I had to downsample it 
     downpcd = pcd.voxel_down_sample(voxel_size=0.05)
     
     # visualise point cloud
     o3d.visualization.draw_geometries([downpcd], zoom=0.3412,
                                   front=[0.4257, -0.2125, -0.8795],
                                   lookat=[2.6172, 2.0475, 1.532],
                                   up=[-0.0694, -0.9768, 0.2024])

# Helper method to call method to load .ply and .npy point cloud files        
def setup():
     pointCloud = loadPointCloud_npy()
     loadPointCloud_ply()
     return pointCloud

# main method
def main():
    #testMethod()
    
    pointCloud = setup() # load point cloud and store in a numpy array
    
    # Cluster the point cloud
    clustering = Clustering(pointCloud)
    clustering.k_means_clustering(2)
            
if __name__=="__main__":
    main()