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
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import open3d.ml.torch as o3d_ml_torch


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

     def silhouette(self):
    
          x = self.pcd
     
          K = range(2,10)
          for k in K:
               fig, (ax1, ax2) = plt.subplots(1, 2)
               fig.set_size_inches(18, 7)

               # The 1st subplot is the silhouette plot
               # The silhouette coefficient can range from -1, 1 but in this example all
               # lie within [-0.1, 1]
               ax1.set_xlim([-0.1, 1])
               # The (n_clusters+1)*10 is for inserting blank space between silhouette
               # plots of individual clusters, to demarcate them clearly.
               ax1.set_ylim([0, len(x) + (k + 1) * 10])

               # Initialize the clusterer with n_clusters value and a random generator
               # seed of 10 for reproducibility.
               clusterer = KMeans(n_clusters= k, random_state=10)
               cluster_labels = clusterer.fit_predict(x)

               silhouette_avg = silhouette_score(x, cluster_labels)
               print(
                    "For n_clusters =",
                         k,
                    "The average silhouette_score is :",
                         silhouette_avg,
               )
               sample_silhouette_values = silhouette_samples(x, cluster_labels)
     
               y_lower = 10
               for i in range(k):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / k)
                    ax1.fill_betweenx(
                         np.arange(y_lower, y_upper),
                         0,
                         ith_cluster_silhouette_values,
                         facecolor=color,
                         edgecolor=color,
                         alpha=0.7,
                    )

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

               ax1.set_title("The silhouette plot for the various clusters.")
               ax1.set_xlabel("The silhouette coefficient values")
               ax1.set_ylabel("Cluster label")

               ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

               ax1.set_yticks([])  # Clear the yaxis labels / ticks
               ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

               # 2nd Plot showing the actual clusters formed
               colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
               ax2.scatter(
                    x[:, 0], x[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
               )

               # Labeling the clusters
               centers = clusterer.cluster_centers_
               # Draw white circles at cluster centers
               ax2.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    marker="o",
                    c="white",
                    alpha=1,
                    s=200,
                    edgecolor="k",
               )

               for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

               ax2.set_title("The visualization of the clustered data.")
               ax2.set_xlabel("Feature space for the 1st feature")
               ax2.set_ylabel("Feature space for the 2nd feature")

               plt.suptitle(
                    "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                    % k,
                    fontsize=14,
                    fontweight="bold",
               )

          plt.show()


     #output
     # For n_clusters = 2 The average silhouette_score is : 0.3889045527426348
     # For n_clusters = 3 The average silhouette_score is : 0.3403178007585951
     # For n_clusters = 4 The average silhouette_score is : 0.31642498946572917
     # For n_clusters = 5 The average silhouette_score is : 0.30732018048203114
     # For n_clusters = 6 The average silhouette_score is : 0.32511128283087165
     # For n_clusters = 7 The average silhouette_score is : 0.326443764304626
     # For n_clusters = 8 The average silhouette_score is : 0.329213106052044
     # For n_clusters = 9 The average silhouette_score is : 0.31500958433214615

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
     path = "/home/leah/Documents/Thesis_Set_UP/CHSEG/church_registered.ply"
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

def stanford():
     path = "./data/stanford"
     #sdata = o3d_ml_torch.datasets.S3DIS(dataset_path=path)
     #print(sdata.)
     #pcd = o3d.io.read_point_cloud(sdata.path)
     #o3d.visualization.draw_geometries([pcd])
     

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
    total_time = end_time - start_time
    print("End Time = ", end_time.strftime("%H:%M:%S"))
    print("Total Time = ", total_time.strftime("%H:%M:%S"))
            
if __name__=="__main__":
    main()