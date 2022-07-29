from PointCloudLoader import PointCloudLoader
from PointCloudUtils import PointCloudUtils
import tkinter as tk
from tkinter import filedialog as fd

def load():
    file_types = [('Point Cloud Files','*.ply *.npy *.las *.xyz *.pcd')]
    file_name = fd.askopenfilename(title="Open a point cloud file", initialdir="./Data", filetypes=file_types)
    print("Selected File:",file_name)
    vis = True
    
    if file_name == '':
        file_path = "./Data/church_registered.ply"
    else:
        file_path = file_name
        print("file ext:", file_name[-4:])
    #init PointCloudLoader    
    pc_loader = PointCloudLoader(file_path)
    
    options = {0: "PLY", 1: "NPY", 2: "LAS", 3: "RDPLY", 4: "PNNPY"}
    try:
        user_input = int(input("\nMenu:"
                               +"\n0 - for PLY"
                               +"\n1 - for NPY"
                               +"\n2 - for LAS"
                               +"\n3 - for raw PLY Downsampled"
                               +"\n4 - for pnet npy"
                               +"\nYour selection [0/1/2/3/4]: "))
        
        #Open3D Visualisation
        if (options.get(user_input)=="PLY"):
            pcd = pc_loader.load_point_cloud_ply(vis)
        
        #PPTK Visualisation
        elif (options.get(user_input)=="NPY"):
            pcd = pc_loader.load_point_cloud_npy(vis)
            
        elif (options.get(user_input)=="LAS"):
            pcd = pc_loader.load_point_cloud_las(vis)
            
        elif (options.get(user_input)=="RDPLY"):
            pcd = pc_loader.load_point_cloud_dsample_ply(vis)
            
        elif (options.get(user_input)=="PNNPY"):
            pcd, points, feats = pc_loader.load_point_cloud_npy_pnet_final(vis)
                    
        else:
            print("Invalid option selected")
    except ValueError:
        print("Invalid Input. Please Enter a number.")
        
def test_dsample():
    putils = PointCloudUtils()
    putils.downsample_pcd(file_path="", downsample_amt=0.5)
    

def main():
    print("Welcome")
    load()
    #test_dsample()
    

if __name__=="__main__":
    main()