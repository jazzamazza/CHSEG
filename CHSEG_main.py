import numpy as np
import open3d as o3d
from datetime import datetime
import laspy as lp
from sklearn import metrics
from Clustering import Clustering
from PointCloudLoader import *
from Metrics import Testing

# Helper method to call method to load point cloud files  
# Returns a PointCloud in a numpy array      
def setup(option, vis):
     if (option == "1"): pointCloud = loadPointCloud_npy(vis) # setup point cloud with raw features 
     elif (option == "2"): pointCloud = loadPointCloudDS_npy(vis)
     elif (option == "3"): pointCloud = loadPointCloud_las(vis) # setup point cloud with Cloud Compare features
     elif (option == "4"): pointCloud =  loadPointCloud_pNet(vis) # setup point cloud with PointNet++ features
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
                         "\n 2 : Point Cloud with Raw Features DOWNSAMPLED"+
                         "\n 3 : Point Cloud with Cloud Compare Features"+
                         "\n 4 : Point Cloud with PointNet++ Features\n")
          if (userInput == "q"): break
          pcd_choice = userInput
          
          # Setup and visualise point cloud based on user input
          userInput = input("\nVisualise Point Cloud (y/n)?")
          if (userInput == "q"): break
          if (userInput=="y"):
               pointCloud = setup(pcd_choice, True)
          else:
               pointCloud = setup(pcd_choice, False)
          clustering = Clustering(pointCloud, pcd_choice)
          test = Testing(pointCloud, pcd_choice)
     
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s) or skip to testing:"+
                              "\n 0 : K-Means Clustering" +
                              "\n 1 : Clustering Method 1"+
                              "\n 2 : Clustering Method 2"+
                              "\n 3 : Clustering Method 3"+
                              "\n 4 : Clustering Method 4"+
                              "\n 5 : Clustering Method 5"+
                              "\n x : skip to testing"+
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               elif (userInput == "0"): clustering.k_means_clustering_faiss(15)
               elif (userInput == "x"): break
               #elif (userInput == "0"): clustering.k_means_clustering(13)
               # elif (userInput == "1"): clustering.kMediods_clustering(15)
               # elif (userInput == "2"): clustering.GMM_clustering()
               # elif (userInput == "3"): clustering.Clarans_clustering()
               # elif (userInput == "4"): clustering.affinity_progpogation_clustering()
               
          
          while (userInput != "r"):
               userInput = input("\nChoose Testing Method(s):"+
                              "\n 0 : Silhouette Coefficent for kmeans" +
                              "\n 1 : DB Index for Kmeans"+
                              "\n r : Restart the Application\n")

               if (userInput == "q"): break
               elif (userInput == "0"): test.silhouette_kmeans()
               elif (userInput == "1"): test.db_index()

               

            
if __name__=="__main__":
     #convertPCD()
#     loadPointCloud_ply()
     application()
