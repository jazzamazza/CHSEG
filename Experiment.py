from Classification import Classification
from PointCloudLoader import PointCloudLoader
from Clustering import Clustering
from PointCloudUtils import PointCloudUtils
from Metrics import Evaluation
import numpy as np
import open3d as o3d
import pptk
import datetime


class Experiment:
    def __init__(self) -> None:
        # Classes
        self.pcutils = PointCloudUtils()
        self.pcloader = None
        self.clustering = None
        self.classification = None
        
        # File Info
        self.file_path = None
        self.file_ext = None
        self.dataset = None
        self.ds = None
        self.ds_amt = None
        
        # PCD
        self.pcd = None
        self.pcd_truth = None
        self.truth_index = None
        self.points = None
        self.intensity = None
        
        # Clustering Info
        self.n_clusters = None
        self.alg = None
        self.cluster_labels = None
        self.unique_clusters = None
        
        # Classification Info
        self.ground_truth = None
        self.pred_ground_truth = None
        self.test_truth = None
        self.truth_labels = None 
        
        # Other
        self.date_today = datetime.date.today()
        self.time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Metrics
        self.classification_metrics = ['f1', 'jaccard', 'precision', 'recall', 'mean_abs', 'mean_sqr']
        self.clustering_metrics = ['sill','db','rand']

    def fix_truth(self, ground_truth):
        for i in range(0, len(ground_truth)):
            #print(truth)
            ground_truth[i][0] = float(round(ground_truth[i][0]))
            if ground_truth[i][0] != float(0) and ground_truth[i][0] != float(1): print(ground_truth[i][0])
        return ground_truth
    
    def load(self, file_path):
        self.pcloader = PointCloudLoader(file_path)
        file_info = self.pcloader.file_info()
        self.file_path = file_info['path']
        self.dataset = file_info['dataset']
        self.file_ext = file_info['filetype']
        self.ds = file_info['downsampled']
        self.ds_amt = file_info['dsamt']
        self.pcd, self.pcd_truth = self.pcloader.load_point_cloud()
        
        self.points = self.pcd_truth[:, :3]
        self.intensity = self.pcd_truth[:, 3:4]
        if self.dataset == "raw" or self.dataset == "cc":
            self.truth_index = 4
        else:
            self.truth_index = 4
        self.ground_truth = self.fix_truth(self.pcd_truth[:,self.truth_index:self.truth_index + 1])
        self.clustering = Clustering(self.pcd, self.pcd_truth, self.dataset)
        self.classification = Classification(self.ground_truth)
        
        # view = pptk.viewer(self.points, self.intensity.flatten(), self.ground_truth.flatten(), debug=True)
        # view.wait()
        # view.close()

    def cluster(self, alg, n_clusters):
        self.alg, self.n_clusters = alg, n_clusters
        if alg == "kmeans":
            self.cluster_labels = self.clustering.k_means_clustering(n_clusters)
        elif alg == "birch":
            self.cluster_labels = self.clustering.birch_clustering(n_clusters)
        elif alg == "cure":
            self.cluster_labels = self.clustering.cure_clustering(n_clusters)
        elif alg == "aggl":
            self.cluster_labels = self.clustering.agglomerative_clustering(n_clusters)
        self.unique_clusters = np.unique(self.cluster_labels)
    
    def classify(self):
        assert (self.cluster_labels is not None)
        self.classification.classify(self.unique_clusters, self.cluster_labels)
        self.pred_ground_truth = self.classification.pred_truth_labels
        assert ((np.array_equal(self.ground_truth, self.pred_ground_truth)) != True)
        self.truth_labels = np.hstack((self.ground_truth, self.pred_ground_truth))
    
    def pick_file(self, use_default_path=True , default_path="./Data/Datasets/CloudCompare/church_registered_ds_0.075_cc_23_feats.las"):
        if use_default_path:
            file_path = default_path
            file_ext = file_path[-4:]
            print("Selected file: ", file_path)
            print("File ext:", file_ext)
            self.file_path = file_path
            self.file_ext = file_ext
        else:
            root_path = "./Data/"
            file = "church_registered"
            if input("Downsampled File? ([y]/n): ") != "n":
                file+="_ds_"
                file += input("Enter DS amount: ")
                if input("CloudCompare File? ([y]/n): ") != "n":
                    root_path += "Datasets/CloudCompare/"
                    file += "_cc_23_feats.las"
                else:
                    file += input("Enter file type: ")       
            file_path = root_path + file
            file_ext = file_path[-4:]
            print("Selected file: ", file_path)
            print("File ext:", file_ext)
            self.file_path = file_path
            self.file_ext = file_ext
        
    def clusters_pred_to_ply(self, algorithm_name="unknown"):
        points = self.points
        intensity = self.intensity
        truth = self.ground_truth
        clusters = self.cluster_labels
        pred_truth = self.pred_ground_truth
        zeros = np.zeros((np.shape(points)[0], 1))

        assert(np.array_equal(truth, self.ground_truth))
        
        normals = np.hstack((intensity, clusters, zeros))
        colors = np.hstack((truth, pred_truth, zeros))
        p, c, n = o3d.utility.Vector3dVector(points), o3d.utility.Vector3dVector(colors), o3d.utility.Vector3dVector(normals)
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors, pcd.normals = p, c, n
        
        data_info = algorithm_name + "_" + self.dataset + "_" + str(self.n_clusters) + "_" + str(self.date_today) + "_" + str(self.time)
        if self.ds:
            data_info = "ds_" + str(self.ds_amt) + "_" + data_info

        o3d.io.write_point_cloud("./Data/Clustered/"+algorithm_name+"/church_registered_clusters_"+data_info+".ply", pcd)

        view = pptk.viewer(
            points, 
            clusters.flatten(), 
            truth.flatten(),
            pred_truth.flatten(), 
            intensity.flatten(), 
            debug=True
        )
        view.wait()
        view.close()
        
    def run_experiment(self):
        self.pick_file(use_default_path=True, default_path="./Data/church_registered_ds_0.075.npy")
        self.load(self.file_path)
        self.cluster("kmeans", 125)
        self.classify()
        self.clusters_pred_to_ply(self.alg)
        evalualtion = Evaluation(self.truth_labels)
        evalualtion.evaluate_classification(self.ground_truth, self.pred_ground_truth, metric_choice="all")
        evalualtion.evaluate_clusters(self.ground_truth, self.pred_ground_truth, self.cluster_labels, self.pcd, metric_choice="all")
    
if __name__ == "__main__":
    my_experiment = Experiment()
    my_experiment.run_experiment()