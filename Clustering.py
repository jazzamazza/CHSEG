from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Jared methods: BIRCH, CURE, AGGLOMERATIVE, (ROCK)
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster.cure import cure
from pyclustering.cluster.rock import rock
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.encoder import type_encoding
from sklearn.neighbors import kneighbors_graph

# from pyclustering.cluster import cluster_visualizer
# from pyclustering.cluster import cluster_visualizer_multidim
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import numpy as np
import pptk
import open3d as o3d

# from tqdm import tqdm

from PointCloudUtils import PointCloudUtils
from Classification import Classification

# Clustering class with various clustering methods
class Clustering:
    def __init__(self, pcd_input, pcd_wtruth, pcd_type=None):
        self.pcd = pcd_input
        self.pcd_truth = pcd_wtruth
        self.pcd_type = pcd_type
        self.pcutils = PointCloudUtils()
        self.truth_index = 4
        self.truth_labels = self.pcd_truth[:, self.truth_index : self.truth_index + 1]
        self.cluster_labels = None
        self.classification = Classification(self.truth_labels)

    def clusters_to_ply(self, clusters, algorithm_name="unknown", truth_labels=None):
        points = self.pcd_truth[:, :3]
        intensity = self.pcd_truth[:, 3:4]
        if truth_labels is None:
            truth = self.pcd_truth[:, 4:5]
        else:
            truth = truth_labels
        zeros = np.zeros((np.shape(points)[0], 1))

        self.pcutils.get_attributes(points, "Points")
        self.pcutils.get_attributes(intensity, "Intensity")
        self.pcutils.get_attributes(truth, "Truth")
        self.pcutils.get_attributes(clusters, "Clusters")

        normals = np.hstack((intensity, clusters, zeros))
        colors = np.hstack((truth, zeros, zeros))

        self.pcutils.get_attributes(normals, "Normals")
        self.pcutils.get_attributes(colors, "Colors")

        p = o3d.utility.Vector3dVector(points)
        n = o3d.utility.Vector3dVector(normals)
        c = o3d.utility.Vector3dVector(colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors, pcd.normals = p, c, n

        o3d.io.write_point_cloud(
            "./Data/Clustered/church_registered_clusters_" + algorithm_name + ".ply",
            pcd,
        )

        # view = pptk.viewer(points, clusters.flatten(), truth.flatten(), intensity.flatten(), debug=True)
        # view.wait()
        # view.close()

    def print_heading(self, title="Clustering"):
        heading = (
            "\n" + ("*" * len(title)) + " " + title + " " + ("*" * len(title)) + "\n"
        )
        print(heading)

    def k_means_clustering(self, k=3, n_init=10):
        self.print_heading("K-Means Clustering")
        # number of clusters (k)
        kmeans = KMeans(n_clusters=k, n_init=n_init)
        print("*!* K-Means Clustering start on", k, "clusters *!*")
        # run kmeans on input data and get labels
        cluster_labels = kmeans.fit_predict(self.pcd)
        cluster_labels = np.vstack((cluster_labels))
        self.cluster_labels = cluster_labels
        print("*!* K-Means Clustering done *!*")
        unique_labels = np.unique(cluster_labels)
        assert len(unique_labels) == k
        # centroids = kmeans.cluster_centers_
        return self.cluster_labels

    def birch_clustering(self, k):
        self.print_heading("BIRCH Clustering")
        print("*!* Using", k, "Clusters *!*")
        birch = Birch(n_clusters=k, threshold=0.4)
        print("-> Fit start")
        birch.fit(self.pcd)
        print("<- Fit end")
        print("-> Pred start")
        cluster_labels = birch.predict(self.pcd)
        self.cluster_labels = cluster_labels
        print("<- Pred end")
        unique_labels = np.unique(cluster_labels)
        assert len(unique_labels) == k
        return self.cluster_labels

    def agglomerative_clustering(self, k, affinity="euclidean", linkage="ward"):
        self.print_heading("Agglomerative Clustering")
        #A = kneighbors_graph(self.pcd, 100, mode='connectivity', include_self=True)
        agg_clustering = AgglomerativeClustering(
            n_clusters=k, affinity=affinity, linkage=linkage, memory="./.cache/", connectivity=None
        )
        print(
            "Starting using:",
            k,
            "clusters, Affinity is:",
            affinity,
            ", Linkage is: ",
            linkage,
        )
        cluster_labels = agg_clustering.fit_predict(self.pcd)
        print("Clustering complete")
        cluster_labels = np.vstack(cluster_labels)
        self.cluster_labels = cluster_labels
        return self.cluster_labels

    def rock_clustering(self, k=3, eps=1.0):
        self.print_heading("ROCK Clustering")
        rock_cluster = rock(self.pcd, eps, k, ccore=True)
        print("Starting using", k, "clusters, and a connectivity radius of", eps)
        rock_cluster.process()
        print("Clustering finished")
        clusters = rock_cluster.get_clusters()
        encoding = rock_cluster.get_cluster_encoding()
        encoder = cluster_encoder(encoding, clusters, self.pcd)
        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
        cluster_labels = np.vstack(np.array(encoder.get_clusters()))
        # self.clusters_to_ply(cluster_labels, "rock")
        self.cluster_labels = cluster_labels
        return self.cluster_labels

    def cure_clustering(self, k=10, reps=5, comp=0.5, ccore=True):
        self.print_heading("CURE Clustering")
        # *!* to do num rep_points, compression *!*
        cure_cluster = cure(self.pcd, k, reps, comp, ccore)
        print("Starting using", k, "clusters")
        cure_cluster.process()
        print("Clustering finished")
        clusters = cure_cluster.get_clusters()
        # means = cure_cluster.get_means()
        # reps = cure_cluster.get_representors()
        encoding = cure_cluster.get_cluster_encoding()
        encoder = cluster_encoder(encoding, clusters, self.pcd)
        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
        cluster_labels = encoder.get_clusters()
        cluster_labels = np.array(cluster_labels)
        cluster_labels = np.vstack(cluster_labels)
        self.clusters_to_ply(cluster_labels, "cure")
        self.cluster_labels = cluster_labels
        return self.cluster_labels
