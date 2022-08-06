from Clustering import Clustering
from PointCloudLoader import PointCloudLoader
from PointCloudUtils import PointCloudUtils
from tkinter import filedialog as fd
import numpy as np
from tkinter import *
      
def setup(pnet):
    """Helper method to call method to load point cloud files. Returns a PointCloud in a numpy array.

    Args:
        pnet (bool): pnet option

    Returns:
        ndarray: PointCloud in a numpy array
    """
    print("###### POINT CLOUD SETUP ######")
    
    gui_choice = input("Select file from GUI:\n Would you like to use a GUI to select the Point Cloud file? [y/n]\n->")
    if gui_choice == 'y':
        gui = True
    else:
        gui = False
    
    vis_choice = input("Visualisation:\n Would you like to visualise the selected Point Cloud? [y/n]\n->")
    if vis_choice == 'y':
        vis = True
    else:
        vis = False
        
    ds_choice = input("Downsampling:\n Would you like to downsample the selected Point Cloud? [y/n]\n->")
    if ds_choice == 'y':
        ds = True
        ds_amt = float(input("Downsampling:\n Enter the downsample amount.\n->"))
        print ("ds amnt =", ds_amt)
    else:
        ds = False  
    
    if(gui):
        print("###### POINT CLOUD LOADER ######")
        root = Tk()
        file_types = (('point cloud files','*.ply *.npy *.las *.xyz *.pcd'),("all files","*.*"))
        root.filename = fd.askopenfilename(title="Select a point cloud file", initialdir="./Data", filetypes=file_types)
        root.withdraw()
        file_name = root.filename
        
        if file_name == '':
            file_path = "./Data/church_registered.npy"
            print("default file selected:", file_path)
        else:
            file_path = file_name
            file_ext = file_name[-4:]
            print("selected file:",file_name)
            print("file ext:", file_ext)
    else:
        #HARD CODE PATH
        file_path = "./Data/church_registered_alt_dsample_0.05.las" #C:\Users\Jared\Code\Thesis\CHSEG\Data\church_registered_alt_dsample_0.05.las
        file_ext = file_path[-4:]
        print("selected file:",file_path)
        print("file ext:", file_ext)
        
    pc_loader = PointCloudLoader(file_path, file_ext)
     
    if(ds):
        pcutils = PointCloudUtils()        
        if file_ext == '.npy':
            ##not normal tp chnage
            ds_path_npy, ds_path_ply = pcutils.npy_raw_alt(file_path, ds_amt) 
            pc_loader = PointCloudLoader(ds_path_npy, file_ext)
        
         
    if(pnet):
        point_cloud = pc_loader.loadPointCloud_pNet(vis) # setup point cloud with PointNet++ features
        np.save(arr=point_cloud, file="./Data/church_registered_pnet.npy")
        return point_cloud, False
        
    elif (file_ext == ".ply"):
        pcd = pc_loader.load_point_cloud_ply(vis)
        return pcd, False
    
    elif (file_ext == ".npy"):
        pcd, pcd_all = pc_loader.load_point_cloud_npy(vis)
       
        truth_choice = input("Include truth label:\n Would you like to include the truth labels in the selected Point Cloud? [y/n]\n->")
        if truth_choice == 'y':
            truth = True
        else:
            truth = False
            
        if(truth):
            return pcd, False
        else:
            return pcd_all, True
        
    elif (file_ext == ".las"):
        pcd = pc_loader.load_point_cloud_las(vis)
        return pcd, False
    
    else:
        print("invalid file")
        exit(1)

# interactive application
def application():
     user_input = ""
     while (user_input != "q"):
          print("--------------Welcome---------------")
          print("Type q to quit the application.")
          print("Choose Loaded Point Cloud Type:")
          user_input = input(" 1 : Load Point Cloud"+
                         "\n 2 : Load Point Cloud PointNet++"+
                         "\n q : or quit the app\n")
          if (user_input == "q"): break
          elif (user_input == '1'):
              point_cloud, truth = setup(False)
          elif (user_input == '2'):
              point_cloud, truth = setup(True)
              
          pcd_choice = user_input
          
          if truth:
            #clustering = Clustering(point_cloud, pcd_with_truths, pcd_choice)
            clustering = Clustering(point_cloud, point_cloud, pcd_choice)
          else:
            clustering = Clustering(point_cloud, point_cloud, pcd_choice) 
          userInput=""
          while (userInput != "r"):
               # cluster point cloud    
               userInput = input("\nChoose Clustering Method(s):"+
                              "\n 0 : K-Means Clustering fais" +
                              "\n 1 : sill"+
                              "\n 2 : Birch"+
                              "\n 3 : cure"+
                              "\n 4 : aprop"+
                              "\n 5 : kmed"+
                              "\n 6 : qual"+
                              "\n q : or quit the app"+
                              "\n r : Restart the Application\n")
               if (userInput == "q"): break
               elif (userInput == "0"): clustering.k_means_clustering_faiss(15, "")
               elif (userInput == "1"): clustering.silhouette()
               elif (userInput == "2"): clustering.birch(13)
               elif (userInput == "3"): clustering.cure_clustering(3)
               elif (userInput == "4"): clustering.affinity_progpogation_clustering()
               elif (userInput == "5"): clustering.kMediods_clustering(14)
               elif (userInput == "6"): clustering.find_quality()
            
if __name__=="__main__":
    application()
