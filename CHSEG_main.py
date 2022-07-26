from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from tkinter import filedialog as fd

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
def setup(option, vis):
     if (option == "1"): pointCloud = loadPointCloud_npy(vis) # setup point cloud with raw features 
     elif (option == "2"): pointCloud = loadPointCloud_las(vis) # setup point cloud with Cloud Compare features
     elif (option == "3"): pointCloud = loadPointCloud_pNet(vis) # setup point cloud with PointNet++ features
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
                         "\n 3 : Point Cloud with PointNet++ Features\n")
          if (userInput == "q"): break
          pcd_choice = userInput
          
          # Setup and visualise point cloud based on user input
          userInput = input("\nVisualise Point Cloud (y/n)?")
          if (userInput == "q"): break
          if (userInput=="y"):
               pointCloud = load(True)
          else:
               pointCloud = load(False)
          clustering = Clustering(pointCloud, pcd_choice)
     
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s):"+
                              "\n 0 : K-Means Clustering" +
                              "\n 1 : Clustering Method 1"+
                              "\n 2 : Clustering Method 2"+
                              "\n 3 : Clustering Method 3"+
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               elif (userInput == "0"): clustering.k_means_clustering_faiss(15, "")
               elif (userInput == "1"): clustering.optics_clustering()
            
if __name__=="__main__":
    application()
