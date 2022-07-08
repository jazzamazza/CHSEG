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
from sklearn_extra.cluster import KMedoids #pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip
#from sklearn_extra.cluster import KMedians
import sklearn_extensions as ske


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
          plt.title('Two clusters of data: K-means')
          plt.savefig('k_means_clusters.png') 
          plt.show()
     
     def kMediods(self, k):
          x = self.pcd
          
          print("\n------------------k means---------------------")
          kmediods = KMedoids(n_clusters=k, n_init=10) # number of clusters (k)
          kmediods.fit(x) # apply k means to dataset
          
          # Visualise K-Means
          y_km = KMedoids.predict(x)
          centroids = KMedoids.cluster_centers_
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
          plt.title('Two clusters of data: K-mediods')
          plt.savefig('k_means_clusters.png') 
          plt.show()
     
     # def kMeddians(self, k):
     #      x = self.pcd
          
     #      print("\n------------------k means---------------------")
     #      kmediods = KMedians(n_clusters=k, n_init=10) # number of clusters (k)
     #      kmediods.fit(x) # apply k means to dataset
          
     #      # Visualise K-Means
     #      y_km = KMedoids.predict(x)
     #      centroids = KMedoids.cluster_centers_
     #      unique_labels = np.unique(y_km)

     #      for i in unique_labels:
     #           plt.scatter(x[y_km == i , 0] , x[y_km == i , 1] , label = i, marker='o', picker=True)
     #      plt.scatter(
     #           centroids[:, 0], centroids[:, 1],
     #           s=100, marker='*',
     #           c='red', edgecolor='black',
     #           label='centroids'
     #      )
     #      #plt.legend()
     #      plt.title('Two clusters of data')
     #      plt.savefig('k_means_clusters.png') 
     #      plt.show()

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

     


# Method to load and visualise a point cloud in a .npy file using open3d
def loadPointCloud_npy(vis):
     print("\nLOAD NPY POINT CLOUD DATA")

     #load point cloud to numpy
     inputPath = "/Users/A102178/Desktop/church_registered.npy" #path to point cloud file
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
     downpcd = pcd.voxel_down_sample(voxel_size=0.5) # downsample pcd
     if (vis):
          o3d.visualization.draw_geometries([downpcd])
     
     pc_points = np.asarray(downpcd.points) # convert pcd points to np array
     pc_features = np.asarray(downpcd.normals) # convert pcd additional features to np array
     pc = np.hstack((pc_points, pc_features)) # concatenate the 2 np arrays
     print("Downsampled Point cloud size: ", pc.size)
     print("0 is:", pc[0])
     
     finalPCD = np.delete(pc, [4,5], 1) # remove info unneccessary for clustering from pcd
     print(finalPCD[0])
     
     return finalPCD

def loadPointCloud_xyz(vis):
     print("\nLOAD PLY CC POINT CLOUD DATA\n")
     input = "/Users/A102178/Desktop/church_registered - Cloud.txt"

     PointCloud = np.loadtxt(input)
     pc = np.asarray(PointCloud)

     print("size of point cloud is:", pc.size)
    

     return pc

# Method to load and visualise a point cloud in a .ply file using open3d
def loadPointCloud_ply(vis):
     print("\nLOAD PLY POINT CLOUD DATA")


     #load point cloud .ply file
     path = "/Users/A102178/Desktop/church_registeredD.ply"
     pcd = o3d.io.read_point_cloud(path)
     
     print(pcd)
     
     downpcd = pcd.voxel_down_sample(voxel_size=0.05)
     
     # visualise point cloud
     o3d.visualization.draw_geometries([downpcd], zoom=0.3412,
                                   front=[0.4257, -0.2125, -0.8795],
                                   lookat=[2.6172, 2.0475, 1.532],
                                   up=[-0.0694, -0.9768, 0.2024])
     pc = np.asarray(downpcd.points)
     print("Downsampled Point cloud size: ", pc.size)
     
     return pc
    
def loadPointCloud_las(vis):
     print("\nLOAD LAS POINT CLOUD DATA\n")
     
     path = "/Users/A102178/Desktop/church_registered_cldCmp6.las"  #church_registered _cldCmp (1).las
     pcd = lp.read(path)

     print("All features:", list(pcd.point_format.dimension_names))
     points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()            #lists x, y, z cooridnates 
     print("points", points)
     
     print("Cloud Compare Features:", list(pcd.point_format.extra_dimension_names))
     planarity = np.vstack(pcd['Planarity (0.049006)'])
     intensity = np.vstack(pcd['NormalX'])
     anisotropy = np.vstack(pcd['Anisotropy (0.049006)'])
     Linearity = np.vstack(pcd['Linearity (0.049006)'])
     SurfaceVariation = np.vstack(pcd['Surface variation (0.049006)'])
     Omnivariance = np.vstack(pcd['Omnivariance (0.049006)'])
     Eigenentropy = np.vstack(pcd['Eigenentropy (0.049006)'])
     
     # print("\nPoints", points)
     # print("\nPlanarity:", planarity)
     # print("\nIntensity:", intensity)
     
     # format using open3d
     pc = o3d.geometry.PointCloud()
     pc.points = o3d.utility.Vector3dVector(points)
     zero = intensity # placeholder
     arr = np.hstack((anisotropy, planarity, Eigenentropy))
     arr1 = np.hstack((Linearity, SurfaceVariation, Omnivariance))
     #arr2 = np.hstack(Eigenentropy, Omnivariance)
     #cloudCompareFeatures = np.hstack((arr, zero)) # form a 3D vector to add to o3d pcd
     #cloudCompareFeatures1 = np.hstack((arr1))
     pc.normals = o3d.utility.Vector3dVector(arr) # store additional features (intensity & planarity) in pc.normals
     pc.colors = o3d.utility.Vector3dVector(arr1)
     print(pc)


     #0visualise point cloud
     downpcd = pc.voxel_down_sample(voxel_size=0.05) # downsample pc
     if (vis): 
          o3d.visualization.draw_geometries([downpcd])
     
     pc_points = np.asarray(downpcd.points) # convert pc points to np array
     pc_features = np.asarray(downpcd.normals) # convert pc additional features to np array
     pc_features1 = np.asarray(downpcd.colors)

     print("pc_points", np.asarray(pc.points))
     print("pc_features", np.asarray(pc.normals))

     print("pc_points downsampled", pc_points)
     print("pc_features downsampled", pc_features)

     finalPCD = np.hstack((pc_points, pc_features, pc_features1)) # concatenate the 2 np arrays - ADDED FEATURES1 - SAYS MISSING VALUES FOR K-MEANS
     print("Downsampled Point cloud size: ", finalPCD.size)  # pc_features1
     print("0 is:", finalPCD[0])


     #remove_nanFinal = finalPCD[np.logical_not(np.isnan(finalPCD))]
     # remove_nanFinal = np.nan_to_num(finalPCD)
     # final = np.hstack(remove_nanFinal)
     
     return finalPCD

# Helper method to call method to load .ply and .npy point cloud files        
def setup():
     pointCloud = loadPointCloud_npy()
     pointCloud_ply = loadPointCloud_ply()
     pointCloud_las = loadPointCloud_las()
     pointCloud_xyz = loadPointCloud_xyz()
     return pointCloud, pointCloud_las, pointCloud_ply, pointCloud_xyz

# main method
def main():  
    start_time = datetime.now()
    print("Start Time = ", start_time.strftime("%H:%M:%S"))
    
    pointCloud, pointCloud_las, pointCloud_ply, pointCloud_xyz = setup() # load point cloud and store in a numpy array
    
    # Cluster the point cloud
    clustering = Clustering(pointCloud)
    clustering.k_means_clustering(15)
    
    clustering_cldCmp = Clustering(pointCloud_las) #pointCloud_las
    clustering_cldCmp.k_means_clustering()

    clustering_cldCmpXYZ = Clustering(pointCloud_xyz)
    clustering_cldCmpXYZ.k_means_clustering(15)
    
    end_time = datetime.now()
    print("End Time = ", end_time.strftime("%H:%M:%S"))

# Helper method to call method to load point cloud files  
# Returns a PointCloud in a numpy array      
def newSetup(option, vis):
     if (option == "1"): pointCloud = loadPointCloud_npy(vis) # setup point cloud with raw features 
     elif (option == "2"): pointCloud = loadPointCloud_ply(vis) # setup point cloud with Cloud Compare features
     elif (option == "3"): pointCloud = loadPointCloud_xyz(vis)
     elif (option == "4"): pointCloud = loadPointCloud_las(vis) # setup point cloud with PointNet++ features
     return pointCloud

# interactive application
def application():
     userInput = ""
     while (userInput != "q"):
          print("--------------Welcome---------------")
          print("Type q to quit the application")
          # Choose Point Cloud
          userInput = input("\nChoose Point Cloud Input:"+
                         "\n 1 : Point Cloud with Raw Features"+
                         "\n 2 : Point Cloud with Cloud Compare Features"+
                         "\n 3 : Point Cloud with Cloud Compare Features using a text file"+
                         "\n 4 : Point Cloud with PointNet++ Features\n")
          if (userInput == "q"): break
          pcd_choice = userInput
          
          # Setup and visualise point cloud based on user input
          userInput = input("\nVisualise Point Cloud (y/n)?")
          if (userInput == "q"): break
          if (userInput=="y"):
               pointCloud = newSetup(pcd_choice, True)
          else:
               pointCloud = newSetup(pcd_choice, False)
          clustering = Clustering(pointCloud)
     
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s):"+
                              "\n 0 : K-Means Clustering" +
                              "\n 1 : Clustering Method 1"+
                              "\n 2 : Clustering Method 2"+
                              "\n 3 : Clustering Method 3"+
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               elif (userInput == "0"): clustering.k_means_clustering(15)
            
if __name__=="__main__":
    application()
    #main()