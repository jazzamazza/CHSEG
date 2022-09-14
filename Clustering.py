# sklearn intel acceleration
from sklearnex import patch_sklearn
patch_sklearn()
# Baseline method:
from sklearn.cluster import KMeans
# Jared methods: BIRCH, CURE, AGGLOMERATIVE, (ROCK)
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster.cure import cure
# Not used library implementation has flaws left for completeness sake
from pyclustering.cluster.rock import rock
# Convert encoding to match sklearn
from pyclustering.cluster.encoder import cluster_encoder
from pyclustering.cluster.encoder import type_encoding
# Preprocessing
from sklearn.neighbors import kneighbors_graph
from pyclustering.utils import timedcall
#other
import numpy as np
import pptk
import open3d as o3d
from PointCloudUtils import PointCloudUtils
from Classification import Classification

class Clustering:
    def __init__(self, pcd_input, pcd_wtruth, pcd_type=None):
        """Clustering class with various clustering methods

        Args:
            pcd_input (ndarray): point cloud data
            pcd_wtruth (ndarray): point cloud data w/truth labels
            pcd_type (str, optional): data set label ("raw", "cc", "pnet"). Defaults to None.
        """
        self.pcd = pcd_input
        self.pcd_truth = pcd_wtruth
        self.pcd_type = pcd_type
        
        # set location of truth label depending on dataset
        if self.pcd_type == "raw" or self.pcd_type == "cc":
            self.truth_index = 4
        elif self.pcd_type == "pnet":
            self.truth_index = 3
        else: # None
            self.truth_index = 4
        # create truth label array
        self.truth_labels = self.pcd_truth[:, self.truth_index : self.truth_index + 1]
        self.classification = Classification(self.truth_labels)
        # set when clustering is called
        self.cluster_labels = None
        self.pcutils = PointCloudUtils()

    def print_heading(self, title="Clustering"):
        """Generates a heading

        Args:
            title (str, optional): heading title. Defaults to "Clustering".
        """
        heading = ("\n" + ("*" * len(title)) +
                    " " + title + " " + 
                    ("*" * len(title)) + "\n")
        print(heading)

    def k_means_clustering(self, k=3, n_init=10):
        """KMeans Clustering

        Args:
            k (int, optional): number of clusters. Defaults to 3.
            n_init (int, optional): sklearn cluster centroid seed init. Defaults to 10.

        Returns:
            ndarray: per point cluster labels
        """
        self.print_heading("K-Means Clustering")
        # init KMeans object
        kmeans = KMeans(n_clusters=k, n_init=n_init)
        print("*!* K-Means Clustering start on", k, "clusters *!*")
        # run kmeans on input data and get labels
        print("-> Fit+Pred start")
        cluster_labels = kmeans.fit_predict(self.pcd)
        print("<- Fit+Pred end")
        # vstack to get correct array dimension n, 1
        cluster_labels = np.vstack((cluster_labels))
        self.cluster_labels = cluster_labels
        print("*!* K-Means Clustering done *!*")
        # list of cluster labels (names) i.e. [1,2,3] for 3 clusters
        unique_labels = np.unique(cluster_labels)
        # follows
        assert len(unique_labels) == k
        # can return centroids with `centroids = kmeans.cluster_centers_`
        return self.cluster_labels

    def birch_clustering(self, k=10):
        """Birch clustering

        Args:
            k (int, optional): number of clusters - k. Defaults to 10.

        Returns:
            ndarray: per point cluster labels
        """
        self.print_heading("BIRCH Clustering")
        print("*!* Using", k, "Clusters *!*")
        #init birch object
        birch = Birch(n_clusters=k, threshold=0.4)
        # run birch on input data and get labels
        print("-> Fit start")
        birch.fit(self.pcd)
        print("<- Fit end")
        print("-> Pred start")
        cluster_labels = birch.predict(self.pcd)
        print("<- Pred end")
        self.cluster_labels = cluster_labels
        # list of cluster labels (names) i.e. [1,2,3] for 3 clusters
        unique_labels = np.unique(cluster_labels)
        # follows
        assert len(unique_labels) == k
        return self.cluster_labels

    def agglomerative_clustering(self, k=10, affinity="euclidean", linkage="ward", compute_neighbours=True):
        """Agglomerative clustering

        Args:
            k (int, optional): n clusters - k. Defaults to 10.
            affinity (str, optional): Metric used to compute the linkage. Defaults to "euclidean".
            linkage (str, optional): linkage criterion to use. Defaults to "ward".
            compute_neighbours (bool, optional): speed up with knn graph. Defaults to True.

        Returns:
            ndarray: per point cluster labels
        """
        self.print_heading("Agglomerative Clustering")
        
        if not compute_neighbours:
            # init aggl clustering object
            agg_clustering = AgglomerativeClustering(n_clusters=k, affinity=affinity,
                                                     linkage=linkage, memory="./.cache/")
        else:
            # use 1% of dataset for n-neighours seed - arbitrary
            neighbours = int(np.shape(self.pcd)[0]*0.01)
            print("n neighbours for graph:",neighbours)
            # precompute neighbour graph knearest nighbours to improve speed of algorithm
            k_graph = kneighbors_graph(self.pcd, neighbours, mode='connectivity', include_self=True)
            # init aggl clustering object - affinity and linkage default euclidean and ward
            # connectivity set to knn graph to improve speed
            agg_clustering = AgglomerativeClustering(n_clusters=k, affinity=affinity,
                                                     linkage=linkage, memory="./.cache/", 
                                                     connectivity=k_graph)
        print("Starting using:", k, "clusters, Affinity is:", affinity,", Linkage is: ",linkage,)
        print("-> Fit+Pred start")
        cluster_labels = agg_clustering.fit_predict(self.pcd)
        print("<- Fit+Pred end")
        # vstack to get correct array dimension n, 1
        cluster_labels = np.vstack(cluster_labels)
        self.cluster_labels = cluster_labels
        return self.cluster_labels

    def cure_clustering(self, k=10, reps=5, comp=0.5, ccore=True, timed = False):
        """CURE Clustering

        Args:
            k (int, optional): n clusters - k. Defaults to 10.
            reps (int, optional): n representors. Defaults to 5.
            comp (float, optional): compression value. Defaults to 0.5.
            ccore (bool, optional): speed up with c implementation. Defaults to True.
            timed (bool, optional): timed run returns run time. Defaults to False.

        Returns:
            ndarray: per cluster points
        """
        self.print_heading("CURE Clustering")
        # init cure object
        cure_cluster = cure(self.pcd, k, reps, comp, ccore)
        print("Starting using", k, "clusters")
        print("-> process start")
        if timed:
            time, _ = timedcall(cure_cluster.process)
        else:
            cure_cluster.process()
        print("-> process end")
        clusters = cure_cluster.get_clusters()
        # can get means and reps - `means = cure_cluster.get_means()` and `reps = cure_cluster.get_representors()`
        
        #set encoding to match sklearn (n, 1)
        encoding = cure_cluster.get_cluster_encoding()
        encoder = cluster_encoder(encoding, clusters, self.pcd)
        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
        cluster_labels = encoder.get_clusters()
        cluster_labels = np.array(cluster_labels)
        cluster_labels = np.vstack(cluster_labels)
        self.cluster_labels = cluster_labels
        if timed:
            return time, self.cluster_labels
        return self.cluster_labels
    
    def rock_clustering(self, k=3, eps=1.0):
        # Not used library implementation has flaws left for completeness sake
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
        self.cluster_labels = cluster_labels
        return self.cluster_labels
