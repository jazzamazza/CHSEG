import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
from mpl_toolkits import mplot3d
from datetime import datetime

# Clustering class with various clustering methods
class Clustering:
     def __init__(self, pointCloud):
          self.pcd = pointCloud
    
     # ELBOW METHOD TO FIND OPTIMAL CLUSTERS 
     def elbow_method(self):

        x = np.array(self.pcd)
        print("\n-----Elbow Method-----")
        distortions = []
        inertias = []
 
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(x)
            distortions.append(kmeanModel.inertia_)

        #PLOTS ELBOW GRAPH 
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')        
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()
        # END OF ELBOW METHOD - SEEMS OPTIMAL IS 3 IF DONE CORRECTLY
          
     # k means clustering method --> clusters a dataset into k (given) clusters
     def k_means_clustering(self, k):
        x = np.array(self.pcd)

        print("\n-----k means-------")
        kmeans = KMeans(n_clusters=3,  n_init=10)
        kmeans.fit(x)
        print(kmeans.labels_)
        print(kmeans.cluster_centers_)

        y_km = kmeans.predict(x)

        # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', y_km)) 
        # plt.scatter(x[:,0], x[:,1], c=colors, marker="o", picker=True)

        #PLOTS K-MEANS GRAPH WITH CLUSTERS AND CENTROIDS
        plt.scatter(
            x[y_km == 0, 0], x[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s',
            label='cluster 1'
        )

        plt.scatter(
            x[y_km == 1, 0], x[y_km == 1, 1],
            s=50, c='orange',
            marker='o',
            label='cluster 2'
        )

        plt.scatter(
            x[y_km == 2, 0], x[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v',
            label='cluster 3'
        )
        #centroid 
        plt.scatter(
            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red',
            label='centroids'
        )

        plt.title('optimal number of clusters of data')
          
          #current_time = datetime.now().strftime("%H:%M:%S")
          #print("saving fig: Current Time = ", current_time)
        plt.savefig('k_means_clusters.png') 
          #current_time = datetime.now().strftime("%H:%M:%S")
          #print("saved fig: Current Time = ", current_time)
          
          #current_time = datetime.now().strftime("%H:%M:%S")
          #print("displaying fig: Current Time = ", current_time)
        plt.show()
          #current_time = datetime.now().strftime("%H:%M:%S")
          #print("displayed fig: Current Time = ", current_time)

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
     inputPath = "/Users/A102178/Desktop/church_registered.npy" #path to point cloud file
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
     path = "Users/A102178/Desktop/church_registered.npy"
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
    #testMethod() #this works
    
    pointCloud = setup() # load point cloud and store in a numpy array
    
    # Cluster the point cloud

    clustering = Clustering(pointCloud)
    #clustering.elbow_method()
    clustering.k_means_clustering(3)
            
if __name__=="__main__":
    main()