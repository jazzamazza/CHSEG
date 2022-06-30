import open3d as o3d
import numpy as np
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, pointCloud):
        self.pcd = pointCloud

    def k_means_clustering(self,k):
        print("\n-----k means-------")
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.pcd)

        print("\nCluster centres:")
        print(kmeans.cluster_centers_)
        print("\nLables:")
        print(kmeans.labels_)
    

    def my_kmeans():

        print("\nLOAD NPY POINT CLOUD DATA")
        inputPath = "/Users/A102178/Desktop/church_registered.npy"
        pointCloud = np.load(inputPath)
        x = np.array(pointCloud)
        plt.scatter(x[:,0],x[:,1], label='True Position')

        # print("\n-----k means-------")
        # kmeans = KMeans(n_clusters=2,  n_init=10)
        # kmeans.fit(x)
        # print(kmeans.labels_)
        # print(kmeans.cluster_centers_)

        # y_km = kmeans.predict(x)

        # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', y_km))
        # plt.scatter(x[:,0], x[:,1], c=colors, marker="o", picker=True)
        # plt.scatter(
        #     kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        #     s=250, marker='*',
        #     c='red', edgecolor='black',
        #     label='centroids'
        # )
        # plt.title('Two clusters of data')
        # plt.show()  

    #     plt.scatter(
    #             x[y_km == 0, 0], x[y_km == 0, 1],
    #             s=50, c='lightgreen',
    #             marker='s', edgecolor='black',
    #             label='cluster 1'
    #     )

    #     plt.scatter(
    #         x[y_km == 1, 0], x[y_km == 1, 1],
    #         s=50, c='orange',
    #         marker='o', edgecolor='black',
    #         label='cluster 2'
    #     )

    #     #plot centroids 
    #     plt.scatter(
    #         kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    #         s=250, marker='*',
    #         c='red', edgecolor='black',
    #         label='centroids'
    #     )
    
    # plt.legend(scatterpoints=1)
    # plt.grid()
    # plt.show()

    def loadPointCloud_npy():
        print("\nLOAD NPY POINT CLOUD DATA")

        #load point cloud into numpy
        inputPath = "/Users/A102178/Desktop/church_registered.npy"
        pointCloud = np.load(inputPath)
        print("point cloud size: ", pointCloud.size)

        #format using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointCloud[:,:3])
        print(pcd)
        o3d.visualization.draw_geometries([pcd])

    def loadPointCloud_ply():

        """ print("load a demo ply pcloud")
        ply_point_cloud = o3d.data.PLYPointCloud()
        pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024]) """




        print("Try load a file")
        my_cloud = open("/Users/A102178/Desktop/church_registered.ply", "r")
        print(my_cloud.name)
        pcd = o3d.io.read_point_cloud(my_cloud.name)
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])

        pcd = o3d.io.read_point_cloud("/Users/A102178/Desktop/church_registered.ply")



    if __name__ == '__main__':
       # loadPointCloud_npy()
       # loadPointCloud_ply()
        my_kmeans()