from curses import raw
from operator import truth
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from mpl_toolkits import mplot3d
from datetime import datetime
from yaml import load
import laspy as lp

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud):
          self.pcd = pointCloud
          
     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
          x = self.pcd
          
          print("\n------------------k means---------------------")
          kmeans = KMeans(n_clusters=k, n_init=10) # number of clusters (k)
          kmeans.fit(x) # apply k means to dataset
          
          # Visualise K-Means
          y_km = kmeans.predict(x)
          centroids = kmeans.cluster_centers_
          unique_labels = np.unique(y_km)
          for i in unique_labels:
               plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
          plt.scatter(
               centroids[:, 0], centroids[:, 1],
               s=100, marker='*',
               c='red', edgecolor='black',
               label='centroids'
          )
          #plt.legend()
          plt.title('Two clusters of data')
          plt.savefig('k_means_clusters.png') 
          plt.show()

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
     pcd.points = o3d.utility.Vector3dVector(pointCloud[:,:3]) # add {x,y,z} points to pcd
     intensities = pointCloud[:,3:4] # add intensity values to pcd
     truthLabels = pointCloud[:,4:5] # add truth labels to pcd
     zero = pointCloud[:,4:5] # placeholder
     arr = np.hstack((intensities, truthLabels))
     rawFeatures = np.hstack((arr, zero)) # form a 3D vector to add to o3d pcd
     pcd.normals = o3d.utility.Vector3dVector(rawFeatures) # store additional features (intensity & truth labels) in pcd.normals
     print(pcd)

     # visualise point cloud
     downpcd = pcd.voxel_down_sample(voxel_size=0.05) # downsample pcd
     o3d.visualization.draw_geometries([downpcd])
     
     pc_points = np.asarray(downpcd.points) # convert pcd points to np array
     pc_features = np.asarray(downpcd.normals) # convert pcd additional features to np array
     pc = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
     print("Downsampled Point cloud size: ", pc.size)
     print("0 is:", pc[0])
     
     finalPCD = np.delete(pc, [4,5], 1) # remove info unneccessary for clustering from pcd
     print(finalPCD[0])
     
     return finalPCD

# Method to load and visualise a point cloud in a .ply file using open3d
def loadPointCloud_ply():
     print("\nLOAD PLY POINT CLOUD DATA")

     #load point cloud .ply file
     #path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered.ply"
     path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered _cldCmp.ply"
     pcd = o3d.io.read_point_cloud(path)
     
     print(pcd)

     #print(np.asarray(pcd.points))
     #print("Has colours:", pcd.has_colors(), np.asarray(pcd.colors)[0])
     #print("Has normals:", pcd.has_normals(), np.asarray(pcd.normals)[0])
     #print("Has points:", pcd.has_points(), np.asarray(pcd.points)[0])
     #print("Has covariances:", pcd.has_covariances())
     
     downpcd = pcd.voxel_down_sample(voxel_size=0.05)
     
     # visualise point cloud
     o3d.visualization.draw_geometries([downpcd], zoom=0.3412,
                                   front=[0.4257, -0.2125, -0.8795],
                                   lookat=[2.6172, 2.0475, 1.532],
                                   up=[-0.0694, -0.9768, 0.2024])
     pc = np.asarray(downpcd.points)
     print("Downsampled Point cloud size: ", pc.size)
     
     return pc

def loadPointCloud_las():
     print("\nLOAD LAS POINT CLOUD DATA\n")
     
     path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered _cldCmp.las"
     pcd = lp.read(path)

     print("All features:", list(pcd.point_format.dimension_names))
     points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()
     
     print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
     planarity = np.vstack(pcd['Planarity (0.049006)'])
     intensity = np.vstack(pcd['NormalX'])
     
     print("\nPoints", points)
     print("\nPlanarity:", planarity)
     print("\nIntensity:", intensity)
     
     # format using open3d
     pc = o3d.geometry.PointCloud()
     pc.points = o3d.utility.Vector3dVector(points)
     zero = planarity # placeholder
     arr = np.hstack((planarity, intensity))
     cloudCompareFeatures = np.hstack((arr, zero)) # form a 3D vector to add to o3d pcd
     pc.normals = o3d.utility.Vector3dVector(cloudCompareFeatures) # store additional features (intensity & planarity) in pc.normals
     print(pc)

     # visualise point cloud
     downpcd = pc.voxel_down_sample(voxel_size=0.05) # downsample pc
     o3d.visualization.draw_geometries([downpcd])
     
     pc_points = np.asarray(downpcd.points) # convert pc points to np array
     pc_features = np.asarray(downpcd.normals) # convert pc additional features to np array
     finalPCD = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
     print("Downsampled Point cloud size: ", finalPCD.size)
     print("0 is:", finalPCD[0])
     
     print(finalPCD[0])
     
     return finalPCD

# Helper method to call method to load .ply and .npy point cloud files        
def setup():
     pointCloud = loadPointCloud_npy()
     #pointCloud = loadPointCloud_ply()
     pointCloud_las = loadPointCloud_las()
     return pointCloud, pointCloud_las

# main method
def main():
    #testMethod() #this works
    
    start_time = datetime.now()
    print("Start Time = ", start_time.strftime("%H:%M:%S"))
    
    pointCloud, pointCloud_las = setup() # load point cloud and store in a numpy array
    
    # Cluster the point cloud
    clustering = Clustering(pointCloud)
    clustering.k_means_clustering(15)
    
    clustering_cldCmp = Clustering(pointCloud_las)
    clustering_cldCmp.k_means_clustering(15)
    
    end_time = datetime.now()
    print("End Time = ", end_time.strftime("%H:%M:%S"))
            
if __name__=="__main__":
    main()