# CHSEG
from Classification import Classification
from PointCloudLoader import PointCloudLoader
from Clustering import Clustering
from PointCloudUtils import PointCloudUtils
from Metrics import Evaluation
# other
import numpy as np
import open3d as o3d
import pandas as pd
import pptk
import datetime
from os.path import exists
import time

class Experiment:
    def __init__(self) -> None:
        """Experiment Class
        """
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
        self.n_points = None

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
        self.time = datetime.datetime.now().strftime("%H_%M%p")
        pd.set_option('max_columns', None)

        # Metrics
        self.classification_metrics = [
            "f1",
            "jaccard",
            "precision",
            "recall",
            "mean_abs",
            "mean_sqr",
        ]
        self.clustering_metrics = ["sill", "db", "rand"]
        self.class_eval = None
        self.clust_eval = None
        
        # pandas
        self.experiment_df = None

    def fix_truth(self, ground_truth):
        """Correct ground truth labels to binary labels of 1 or 0.

        Args:
            ground_truth (ndarray): original ground truth labels.

        Returns:
            ndarray: fixed ground truth labels
        """
        for i in range(0, len(ground_truth)):
            # check labels
            ground_truth[i][0] = float(round(ground_truth[i][0]))
            if ground_truth[i][0] != float(0) and ground_truth[i][0] != float(1):
                print(ground_truth[i][0])
        return ground_truth

    def load(self, file_path):
        """Load file.

        Args:
            file_path (str): File path
        """
        # load file and set globals
        self.pcloader = PointCloudLoader(file_path)
        file_info = self.pcloader.file_info()
        self.file_path = file_info["path"]
        self.dataset = file_info["dataset"]
        self.file_ext = file_info["filetype"]
        self.ds = file_info["downsampled"]
        self.ds_amt = file_info["dsamt"]
        # return pcd
        self.pcd, self.pcd_truth = self.pcloader.load_point_cloud()
        self.points = self.pcd_truth[:, :3]
        self.n_points = np.shape(self.points)[0]
        self.intensity = self.pcd_truth[:, 3:4]
        if self.dataset == "raw" or self.dataset == "cc":
            print(self.dataset)
            self.truth_index = 4
        elif self.dataset == "pnet":
            self.truth_index = 3
            print(self.truth_index)
        # fix truths
        self.ground_truth = self.fix_truth(self.pcd_truth[:, self.truth_index : self.truth_index + 1])
        # init objects
        self.clustering = Clustering(self.pcd, self.pcd_truth, self.dataset)
        self.classification = Classification(self.ground_truth)

    def cluster(self, alg, n_clusters):
        """Run clustering algorithm.

        Args:
            alg (str): Algorithm to run. e.g. "kmeans".
            n_clusters (int): n_clusters to cluster on.
        """
        # set globals
        self.alg, self.n_clusters = alg, n_clusters
        # run algorithm and get per point clusters
        if alg == "kmeans":
            self.cluster_labels = self.clustering.k_means_clustering(n_clusters)
            # check dims (n, 1)
            if np.ndim(self.cluster_labels) != 2:
                self.cluster_labels = np.vstack(self.cluster_labels)
        elif alg == "birch":
            self.cluster_labels = self.clustering.birch_clustering(n_clusters)
            # check dims (n, 1)
            if np.ndim(self.cluster_labels) != 2:
                self.cluster_labels = np.vstack(self.cluster_labels)
        elif alg == "cure":
            # params found through experimentation see Tools.py
            self.cluster_labels = self.clustering.cure_clustering(
                n_clusters, reps=40, comp=0.3, ccore=True
            )
            # check dims (n, 1)
            if np.ndim(self.cluster_labels) != 2:
                self.cluster_labels = np.vstack(self.cluster_labels)
        elif alg == "aggl":
            self.cluster_labels = self.clustering.agglomerative_clustering(n_clusters)
            # check dims (n, 1)
            if np.ndim(self.cluster_labels) != 2:
                self.cluster_labels = np.vstack(self.cluster_labels)
        # list of unique labels
        self.unique_clusters = np.unique(self.cluster_labels)

    def classify(self):
        """Run binary classification and get pred truth labels.
        """
        # check clustering has happened
        assert self.cluster_labels is not None
        # run classification
        self.classification.classify(self.unique_clusters, self.cluster_labels)
        self.pred_ground_truth = self.classification.pred_truth_labels
        # check pred and ground truths are not equal
        assert (not np.array_equal(self.ground_truth, self.pred_ground_truth))
        # save labels
        self.truth_labels = np.hstack((self.ground_truth, self.pred_ground_truth))

    def pick_file(
        self,
        use_default_path=True,
        default_path="./Data/Datasets/CloudCompare/church_registered_ds_0.075_cc_23_feats.las",
    ):
        """Pick file to use for experiment.

        Args:
            use_default_path (bool, optional): Defaults to True.
            default_path (str, optional): Alternative path. Defaults to "./Data/Datasets/CloudCompare/church_registered_ds_0.075_cc_23_feats.las".
        """
        if use_default_path:
            # use defualt path
            file_path = default_path
            file_ext = file_path[-4:]
            print("Selected file: ", file_path)
            print("File ext:", file_ext)
            self.file_path = file_path
            self.file_ext = file_ext
        else:
            # otherwise pick file using picker
            root_path = "./Data/"
            file = "church_registered"
            if input("Downsampled File? ([y]/n): ") != "n":
                file += "_ds_"
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
        """Save cluster and pred and ground truth labels to a .ply file to visualise in Open3D.

        Args:
            algorithm_name (str, optional): Algorithm name e.g. "birch". Defaults to "unknown".
        """
        points = self.points
        intensity = self.intensity
        truth = self.ground_truth
        clusters = self.cluster_labels
        pred_truth = self.pred_ground_truth
        zeros = np.zeros((np.shape(points)[0], 1))
        # check labels
        assert np.array_equal(truth, self.ground_truth)

        normals = np.hstack((intensity, clusters, zeros))
        colors = np.hstack((truth, pred_truth, zeros))
        p, c, n = (
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3dVector(colors),
            o3d.utility.Vector3dVector(normals),
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors, pcd.normals = p, c, n

        data_info = (
            algorithm_name
            + "_"
            + self.dataset
            + "_"
            + str(self.n_clusters)
            + "_"
            + str(self.date_today)
            + "_"
            + str(self.time)
        )
        if self.ds:
            data_info = "ds_" + str(self.ds_amt) + "_" + data_info
        file_path = str(
            "./Data/Clustered/"
            + algorithm_name
            + "/church_registered_clusters_"
            + data_info
            + ".ply"
        )
        o3d.io.write_point_cloud(file_path, pcd)
        
        # labels = []
        # truth_1d, pred_1d, clusters_1d = truth.flatten(), pred_truth.flatten(), clusters.flatten()
        # labels = labels.append(truth_1d)
        # labels = labels.append(pred_1d)
        # labels = labels.append(clusters_1d)
        # view = pptk.viewer(points, debug = True)
        # view.attributes(truth_1d, pred_1d, clusters_1d)
        # view.set(point_size=0.025)
        # view.wait()
        self.vis_clusters_pred(points, clusters, truth, pred_truth)

    def vis_clusters_pred(self, points, clusters, truth, pred_truth):
        """View clusters and truth labels using PPTK.

        Args:
            points (ndarray): points
            clusters (ndarray): cluster labels
            truth (ndarray): ground truth labels
            pred_truth (ndarray): pred truth labels
            intensity (ndarray): intensity labels
        """
        view = pptk.viewer(
            points,
            clusters.flatten(),
            truth.flatten(),
            pred_truth.flatten(),
            #debug=True,
        )
        view.set(point_size=0.0075)
        view.set(lookat=[-2, 2, 0], r=50, theta=(np.pi/4))
        time.sleep(1)
        file_start = "./Results/"+self.alg+"_"+self.dataset+"_"
        view.capture((file_start+"clusters"+"_"+str(self.n_clusters)+".png"))
        view.set(curr_attribute_id = 1)
        time.sleep(1)
        view.capture((file_start+"truth"+"_"+str(self.n_clusters)+".png"))
        view.set(curr_attribute_id = 2)
        time.sleep(1)
        view.capture((file_start+"pred"+"_"+str(self.n_clusters)+".png"))
        view.set(curr_attribute_id = 0)
        poses = []
        poses.append([0, 0, 0, 0 * np.pi/2, np.pi/4, 35])
        poses.append([0, 0, 0, 1 * np.pi/2, np.pi/4, 35])
        poses.append([0, 0, 0, 2 * np.pi/2, np.pi/4, 35])
        poses.append([0, 0, 0, 3 * np.pi/2, np.pi/4, 35])
        poses.append([0, 0, 0, 4 * np.pi/2, np.pi/4, 35])
        # time.sleep(10)
        view.play(poses, 2 * np.arange(5), repeat=True, interp='linear')
        time.sleep(6)
        view.set(curr_attribute_id = 1)
        time.sleep(6)
        view.set(curr_attribute_id = 2)
        #view.wait()
        time.sleep(6)
        view.close()

    def create_pandas(self, output_file):
        """Create pandas data frame.

        Args:
            output_file (str): path to output .csv file
        """
        if exists(output_file):
            # check if file exists for this alg, dataset and downsample amount.
            self.experiment_df = pd.read_csv(output_file, sep=',', header=0, index_col=0)
        else:      
            columns = [
                "date",
                "time",
                "data_set",
                "is_down_sample",
                "down_sample_amount",
                "n_points",
                "n_clusters",
                "clustering_algorithm",
            ]
            columns.append("classification_metrics")
            for metric in self.classification_metrics:
                columns.append(metric)
            columns.append("clustering_metrics")
            for metric in self.clustering_metrics:
                columns.append(metric)
            self.experiment_df = pd.DataFrame(data=None, columns=columns)

    def experiment_to_pandas(self, index, output_file="./Results/test_x.csv"):
        """Create data frame for experiment run

        Args:
            index (int): Run index.
            output_file (str, optional): output .csv file. Defaults to "./Results/test_x.csv".
        """
        data = {}
        data["date"] = str(self.date_today)
        data["time"] = str(self.time)
        data["data_set"] = self.dataset
        data["is_down_sample"] = self.ds
        data["down_sample_amount"] = self.ds_amt
        data["n_points"] = self.n_points
        data["n_clusters"] = self.n_clusters
        data["clustering_algorithm"] = self.alg
        data["classification_metrics"] = str(self.classification_metrics)
        for metric in self.classification_metrics:
            data[metric] = self.class_eval[metric]
        data["clustering_metrics"] = str(self.clustering_metrics)
        for metric in self.clustering_metrics:
            data[metric] = self.clust_eval[metric]

        self.experiment_df = self.experiment_df.append(data, ignore_index=True)
        print("\nPandas DataFrame Tail:")
        print(self.experiment_df.tail(5))
        print("End Pandas DataFrame Tail\n")
        # write out run
        self.experiment_writer(output_file)

    def experiment_writer(self, output_file):
        """Write pandas data frame of experiment runs to .csv file.

        Args:
            output_file (str): output .csv file.
        """
        # check if file exists
        if exists(output_file):
            # append to file
            self.experiment_df.tail(1).to_csv(output_file, mode='a', header=False, index=True)
        else:
            # create file
            print("Creating CSV")
            self.experiment_df.to_csv(output_file, mode = 'w', header=True)
        

    def run_experiment(
        self,
        cluster_start,
        cluster_end,
        algs=["kmeans","birch","aggl","cure"],
        data_set_paths=["./Data/PNet/church_registered_ds_0.05.npy"],
        test_out_file = "./Results/test_default.csv"
    ):
        """Cnetral experiment function. Run an experiment.

        Args:
            cluster_start (int): nclusters start value
            cluster_end (int): nclusters end value
            algs (list, optional): list of algorithms to run. Defaults to ["kmeans","birch","aggl","cure"].
            data_set_paths (list, optional): list of datasets to use for experiment. Defaults to ["./Data/PNet/church_registered_ds_0.05.npy"].
            test_out_file (str, optional): .csv file to write results to. Defaults to "./Results/test_default.csv".
        """
        index = 0
        # default metrics
        self.classification_metrics = [
            "f1",
            "jaccard",
            "precision",
            "recall",
            "mean_abs",
            "mean_sqr"]
        self.clustering_metrics = [
            #"sill", 
            "db", 
            "rand"]
        # create eval object
        evaluation = Evaluation(self.truth_labels)
        # create data frame
        self.create_pandas(test_out_file)
        for path in data_set_paths:
            # select input data file
            print("Path:", path)
            self.pick_file(use_default_path=True, default_path=path)
            self.load(self.file_path)
            for k in range(cluster_start, cluster_end + 1):
                # run alg on k clusters
                print("Clusters:", k)
                for alg in algs:
                    # run alg
                    print("Alg:", alg)
                    # save experiment data
                    self.date_today = datetime.date.today()
                    self.time = datetime.datetime.now().strftime("%H_%M%p")
                    # cluster, classify, save pcd.
                    self.cluster(alg, k)
                    self.classify()
                    self.clusters_pred_to_ply(self.alg)
                    # eval classification
                    print("\nEvalute Classification")
                    self.class_eval = evaluation.evaluate_classification(
                        self.ground_truth,
                        self.pred_ground_truth,
                        self.classification_metrics,
                        metric_choice="all",
                    )
                    # eval clustering
                    print("\nEvalute Clustering")
                    self.clust_eval = evaluation.evaluate_clusters(
                        self.ground_truth,
                        self.pred_ground_truth,
                        self.cluster_labels,
                        self.pcd,
                        self.clustering_metrics,
                        metric_choice="all",
                    )
                    # save results to file
                    self.experiment_to_pandas(index, test_out_file)
                    print("iteration:", index)
                    index += 1

# can run from main:
# if __name__ == "__main__":
#     my_experiment = Experiment()
#     my_experiment.run_experiment(10, 50)
