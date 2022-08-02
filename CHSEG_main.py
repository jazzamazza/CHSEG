from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from tkinter import filedialog as fd
from LeahClustering import LeahClustering

def load(vis):
    file_types = [('Point Cloud Files','*.ply *.npy *.las *.xyz *.pcd')]
    file_name = fd.askopenfilename(title="Open a point cloud file", initialdir="./Data", filetypes=file_types)
    print("Selected File:",file_name)
    if file_name == '':
        file_path = "./Data/church_registered.ply"
    else:
        file_path = file_name
    #init PointCloudLoader    
    pc_loader = PointCloudLoader(file_path)
    
    options = {0: "PLY", 1: "NPY", 2: "LAS"}
    try:
        user_input = int(input("\nMenu:\n0 - for PLY\n1 - for NPY\n2 - for LAS\nYour selection [0/1/2]: "))
        
        #Open3D Visualisation
        if (options.get(user_input)=="PLY"):
            pcd = pc_loader.load_point_cloud_ply(vis)
            return pcd
        #PPTK Visualisation
        elif (options.get(user_input)=="NPY"):
            pcd = pc_loader.load_point_cloud_npy(vis)
            return pcd
        elif (options.get(user_input)=="LAS"):
            pcd = pc_loader.load_point_cloud_las(vis)
            return pcd        
        else:
            print("Invalid option selected")
    except ValueError:
        print("Invalid Input. Please Enter a number.")

# Helper method to call method to load point cloud files  
# Returns a PointCloud in a numpy array      
def setup(option, vis, ds, dsSize):
    #SET PATH
    file_path = "Data\church_registered.npy"
    pc_loader = PointCloudLoader(file_path)
    if (option == "1"): 
        pointCloud, pcd_with_truths = pc_loader.load_point_cloud_npy(vis, ds, dsSize) # setup point cloud with raw features 
    elif (option == "2"): 
        pointCloud = pc_loader.load_point_cloud_las(vis, ds, dsSize) # setup point cloud with Cloud Compare features
    elif (option == "3"):
        pointCloud = pc_loader.load_point_cloud_pNet_npy(vis, ds, dsSize)
    elif (option == "4"): 
        pointCloud = pc_loader.loadPointCloud_pNet(vis) # setup point cloud with PointNet++ features
    
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
          clustering = Clustering(pointCloud, pcd_with_truths , pcd_choice)
          leah = LeahClustering(pointCloud, pcd_with_truths , pcd_choice)
     
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s):"+
                              "\n 0 : K-Means Clustering" +
                              "\n 1 : Clustering Method 1"+
                              "\n 2 : Clustering Method 2"+
                              "\n 3 : Clustering Method 3"+
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               #elif (userInput == "0"): clustering.k_means_clustering_faiss(15, "")
               elif (userInput == "0"): clustering.k_means_clustering(13)
               elif (userInput == "1"): leah.dbscan_clustering()
               elif (userInput == "2"): leah.optics_clustering()
               elif (userInput == "2"): leah.mean_shift_clustering()
            
if __name__=="__main__":
    application()
