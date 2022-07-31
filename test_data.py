from PointCloudLoader import PointCloudLoader
from PointCloudUtils import PointCloudUtils
from tkinter import filedialog as fd
import tkinter as tk
#import cure_test as ct


def init_pcl(vis = True):
    #init PointCloudLoader    
    pc_loader = PointCloudLoader()
    
    options = {0: "PLY", 1: "NPY", 2: "LAS", 3: "RDPLY", 4: "PNNPY"}
    try:
        user_input = int(input("\nMenu:"
                               +"\n0 - for PLY"
                               +"\n1 - for NPY"
                               +"\n2 - for LAS"
                               +"\n3 - for raw PLY Downsampled"
                               +"\n4 - for pnet npy"
                               +"\nYour selection [0/1/2/3/4]: "))
        
        if (options.get(user_input)=="PLY"):
            pcd = pc_loader.load_point_cloud_ply(vis)
            
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
            exit(1)
    
    except ValueError:
        print("Invalid Input. Please Enter a number.")
        exit(1)   

def load():
    file_types = [('Point Cloud Files','*.ply *.npy *.las *.xyz *.pcd')]
    file_name = fd.askopenfilename(title="Open a point cloud file", initialdir="./Data", filetypes=file_types)
    print("Selected File:",file_name)
    
    if file_name == '':
        file_path = "./Data/church_registered.ply"
    else:
        file_path = file_name
        print("file ext:", file_name[-4:])
        
    return file_path

        
def test_dsample():
    f_path = load()
    putils = PointCloudUtils()
    putils.downsample_pcd(file_path=f_path, downsample_amt=0.03)
    

def main():
    print("Welcome")
    test_dsample()
    #ct.cluster_sample1()
    

if __name__=="__main__":
    main()