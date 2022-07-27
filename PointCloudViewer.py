import pptk
import open3d as o3d
import numpy as np

class PointCloudViewer:
    """PointCloudViewer for viewing PointClouds
    """
    def __init__(self, viewer = "default", downsample_o3d = 0):
        self.viewer = viewer
        self.downsample_o3d = downsample_o3d
        
    def vis_npy(self, points, intensity, truth_label):
        options = {0: "O3D", 1: "PPTK"}
        try:
            user_input = int(input("\nVisualisation Menu:\n0 - for Open3D\n1 - for PPTK\nYour selection [0/1]: "))
            
            #Open3D Visualisation
            if (options.get(user_input)=="O3D"):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points) # add {x,y,z} points to pcd
                #build (n,3) vector to store in normals
                zero = truth_label # placeholder
                raw_features = np.hstack((intensity, truth_label, zero)) # form a 3D vector to add to o3d pcd
                pcd.normals = o3d.utility.Vector3dVector(raw_features) # store additional features (intensity & truth labels) in pcd.normals
                print(pcd)
                
                if (self.downsample_o3d > 0):
                    downpcd = pcd.voxel_down_sample(voxel_size=self.downsample_o3d) # downsample pcd
                    o3d.visualization.draw_geometries([downpcd])
                else:
                    o3d.visualization.draw_geometries([pcd])
            
            #PPTK Visualisation
            elif (options.get(user_input)=="PPTK"):
                print("Visualising in PPTK")
                intensity_1d = intensity.flatten()
                truth_label_1d = truth_label.flatten()
                view = pptk.viewer(points,intensity_1d, truth_label_1d)
                print("PPTK Loaded")
                
            else:
                print("Invalid option selected")
        except ValueError:
            print("Invalid Input. Please Enter a number.")
            
    def vis_ply(self, pcd):
        if (self.downsample_o3d > 0):
            downpcd = pcd.voxel_down_sample(voxel_size=self.downsample_o3d) # downsample pcd
            o3d.visualization.draw_geometries([downpcd])
        else:
            o3d.visualization.draw_geometries([pcd])