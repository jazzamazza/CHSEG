from PointCloudLoader import PointCloudLoader
from Clustering import Clustering


class Experiment:
    def __init__(self) -> None:
        self.dataset = None
        self.ds = None
        # self.alg
        self.pcloader = None
        self.pcd = None
        self.pcd_truth = None
        self.clustering = None
        self.cluster_labels = None

    def load(self, file_path, dataset):
        self.pcloader = PointCloudLoader(file_path)
        self.dataset = dataset
        self.pcd, self.pcd_truth = self.pcloader.load_point_cloud()

    def cluster(self, alg, n_clusters):
        self.clustering = Clustering(self.pcd, self.pcd_truth, self.dataset)
        if alg == "kmeans":
            self.cluster_labels = self.clustering.k_means_clustering(n_clusters)
        elif alg == "birch":
            self.cluster_labels = self.clustering.birch_clustering(n_clusters)
