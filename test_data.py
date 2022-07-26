from PointCloudLoader import PointCloudLoader
import tkinter as tk
from tkinter import filedialog as fd

def load():
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
            pcd = pc_loader.load_point_cloud_ply(True)
        
        #PPTK Visualisation
        elif (options.get(user_input)=="NPY"):
            pcd = pc_loader.load_point_cloud_npy(True)
            
        elif (options.get(user_input)=="LAS"):
            pcd = pc_loader.load_point_cloud_las(True)
                    
        else:
            print("Invalid option selected")
    except ValueError:
        print("Invalid Input. Please Enter a number.")

def main():
    print("Welcome")
    load()
    

if __name__=="__main__":
    main()