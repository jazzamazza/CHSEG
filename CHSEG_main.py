from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from tkinter import filedialog as fd
from Metrics import Testing


# Helper method to call method to load point cloud files  
# Returns a PointCloud in a numpy array      
def setup(option, vis, ds, dsSize):
    #SET PATH
    #file_path = "/content/drive/Shareddrives/CHSEG/data/church_registered _cloudCompare.las"
    file_path =  "/Users/A102178/Desktop/data/church_registered.npy"
    #pc_loader = PointCloudLoader(file_path1)
    if (option == "1"): 
        pc_loader = PointCloudLoader(file_path)
        pointCloud, pcd_with_truths = pc_loader.load_point_cloud_npy(vis, ds, dsSize) # setup point cloud with raw features 
    elif (option == "2"): 
        pc_loader = PointCloudLoader(file_path)
        pointCloud, pcd_with_truths = pc_loader.load_point_cloud_las(vis, ds, dsSize) # setup point cloud with Cloud Compare features
    elif (option == "3"):
        pc_loader = PointCloudLoader(file_path)
        pointCloud = pc_loader.load_point_cloud_pNet_npy(vis, ds, dsSize)
    elif (option == "4"): 
        pc_loader = PointCloudLoader(file_path)
        pointCloud = pc_loader.loadPointCloud_pNet(vis) # setup point cloud with PointNet++ features
 
    #return pointCloud
    return pointCloud, pcd_with_truths

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
                         "\n 3 : Point Cloud with PointNet++ Features"+
                         "\n 4 : Create Point Cloud with PointNet++ Features\n")
                      
          if (userInput == "q"): break

          pcd_choice = userInput
          vis, ds = False, False
          dsSize = 0
          
          # Setup and visualise point cloud based on user input
          userInput = input("\nVisualise Point Cloud (y/n)?")
          if (userInput == "q"): break
          elif (userInput=="y"): vis = True
          
          if (pcd_choice!="4"):
            userInput = input("\nDownsample Point Cloud (y/n)?")
            if (userInput == "q"): break
            elif (userInput=="y"): 
                ds = True
                userInput = input("\nSpecify Downsample Size (0.5, 1, 2, etc.)?")
                if (userInput == "q"): break
                dsSize = float(userInput)

          pointCloud, pcd_with_truths = setup(pcd_choice,vis, ds, dsSize)
          #pointCloud = setup(pcd_choice,vis, ds, dsSize)
          clustering = Clustering(pointCloud, pcd_with_truths , pcd_choice)
          test = Testing(pointCloud, pcd_choice)
     
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s):"+
                              "\n 0 : K-Means Clustering" +
                              "\n 1 : Clustering Method 1"+
                              "\n 2 : Clustering Method 2"+
                              "\n 3 : Clustering Method 3"+
                              "\n 4 : Clustering Method 4"+
                              "\n 5 : Clustering Method 5"+
                              "\n 6 : Clustering Method 6"+
                              "\n 7 : Clustering Method 7"+
                              "\n x : skip to testing"+
                              "\n r : Restart the Application\n")
               
               if (userInput == "x"): break
               elif (userInput == "0"): clustering.k_means_clustering(13)
               elif (userInput == "1"): clustering.kMediods_clustering(15)
               elif (userInput == "2"): clustering.GMM_clustering()
               elif (userInput == "3"): clustering.Clarans_clustering()
               elif (userInput == "4"): clustering.affinity_progpogation_clustering()
               elif (userInput == "5"): clustering.KMedians_clustering()
               elif (userInput == "6"): clustering.fuzzy_cmeans_clustering()
               elif (userInput == "7"): clustering.tryfuzzy()

          while (userInput != "r"):
               userInput = input("\nChoose Testing Method(s):"+
                              "\n 0 : Silhouette Coefficent for Kmeans" +
                              "\n 1 : Silhouette Coefficent for GMM" +
                              "\n 2 : Silhouette Coefficent for Kmedians" +
                              "\n 3 : Silhouette Coefficent for CLARANS" +
                              "\n 4 : Silhouette Index for fuzzy c-means"+
                              "\n 5 : DB Index for kmeans"+
                              "\n 6 : DB Index for GMM"+
                              "\n 7 : DB Index for kmedians"+
                              "\n 8 : DB Index for fuzzy c-means"+
                              "\n 9 : BIC"+
                              "\n x : Skip to Classification metrics" +
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               elif (userInput == "x"): test.classification_metrics()
               elif (userInput == "0"): test.silhouette_kmeans()
               elif (userInput == "1"): test.silhouette_GMM()
               elif (userInput == "2"): test.silhouette_kmedians()
               elif (userInput == "3"): test.silhouette_Clarans()
               elif (userInput == "4"): test.silhouette_fuzzy_cmeans()
               elif (userInput == "5"): test.db_index_kmeans()
               elif (userInput == "6"): test.db_index_GMM()
               elif (userInput == "7"): test.db_index_Kmedians()
               elif (userInput == "8"): test.db_index_fuzzy_cmeans()
               elif (userInput == "9"): test.BIC()
            
if __name__=="__main__":
    application()
