import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from plyfile import PlyData, PlyElement

class Clustering:
    def k_means():
        print("kmeans")

'''IO Class'''
class IO:
    def __init__(self, data_path):
        self.data_path = data_path


    def load_point_cloud_npy(self, file):
        print("Load Point Cloud Data:")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        self.pcd = np.load(self.data_path / file)
        dims = self.pcd[1].size
        size = self.pcd.size
        #print(pcd)
        #print(dims)
        print("Loaded... size is: ", size,
        "... dimensions are: ", dims)
        #return self.pcd
        #x,y,z,intesity,label = np.array_split(pcd,5)
        #print("Split")

        #for point in pcd:
        #    print(point[2])
        #    for p in point:
        #        print(p)
        

    def load_point_cloud_ply(self, file):
        print("Load Point Cloud Data:")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        path = self.data_path / file

        pcd = o3d.io.read_point_cloud(path, print_progress=True)
        #print(pcd)
        print("Loaded...")
        print(np.asarray(pcd.points))
        print(np.asarray(pcd.covariances))

        """ read XYZ point cloud from filename PLY file """
        """ plydata = PlyData.read(self.data_path+file)
        pc = plydata['vertex'].data
        pc_array = np.array([[x, y, z] for x,y,z in pc]) """
        # return pc_array

        # my laptop cannot render the normal point cloud, so I had to downsample it 
        #downpcd = pcd.voxel_down_sample(voxel_size=0.05)

    def point_in_point_cloud(self, index):
        x = self.pcd[index][0]
        print("x:",x)




def main():
    io = IO(Path("./data/"))
    #io.load_point_cloud_ply("church_registered.ply")
    io.load_point_cloud_npy("church_registered.npy")
    io.point_in_point_cloud(1)
    

if __name__ == "__main__":
    main()