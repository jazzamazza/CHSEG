import open3d as o3d
import numpy as np
import pptk

def visTest():
    print("\nLOAD NPY POINT CLOUD DATA DOWNSAMPLED")

     #load point cloud to numpy
    inputPath = "/Users/jaredmay/Downloads/TEST.npy" #path to point cloud file
    pointCloud = np.load(inputPath)
    print("Point cloud size: ", pointCloud.size)
    print("point cloud shape: ", pointCloud.shape)
    print("point cloud", pointCloud)
    
    view = pptk.viewer(pointCloud)
    view.set(point_size=0.005)

    # format using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCloud)
    print(pcd)
    o3d.visualization.draw_geometries([pcd])

def visPCD():

    print('pptk test')

    #load point cloud .ply file
    path = "/Volumes/Space120/Datasets/UCT/Data/church_registered.ply"
    #path = "/Volumes/Space120/Datasets/UCT/Data/church_registered_downsampled_0.05.ply"
    pcd = o3d.io.read_point_cloud(path)
    print(pcd)

    pcd_npy = np.asarray(pcd.points)
    print('npy points: ', pcd_npy)
    print("Point cloud size: ", pcd_npy.size)
    print("point cloud shape: ", pcd_npy.shape)
    #print("point cloud", pointCloud)

    pcd_npy_col = np.asarray(pcd.colors)
    print('npy points colours: ', pcd_npy_col)

    # if (pcd.has_normals()):
    #     pcd_npy_norm = np.asarray(pcd.normals)
    #     print('npy points norms: ', pcd_npy_norm)

    pcd_npy_norm = np.asarray(pcd.normals)
    print('npy points norms: ', pcd_npy_norm)

    view = pptk.viewer(pcd_npy, pcd_npy_col, pcd_npy_norm)
    #view = pptk.viewer(pcd_npy, pcd_npy_col)
    view.set(point_size=0.005)
        
def main():
    user_input = input("y/n?\n")
    if (user_input == 'y'):
        visPCD()
    elif(user_input == 'n'):
        visTest()

if __name__=="__main__":
    main()
