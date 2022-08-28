import numpy as np
from tqdm import tqdm


class Classification:
    def __init__(self, pcd_input, pcd_wtruth, pcd_type=None):
        self.pcd = pcd_input
        self.pcd_truth = pcd_wtruth
        self.pcd_type = pcd_type

        if self.pcd_type == None:
            self.truth_index = 4
        elif self.pcd_type == "raw":
            self.truth_index = 4
        elif self.pcd_type == "cc":
            self.truth_index = 4
        else:
            self.truth_index = 4

        self.truth_labels = self.pcd_truth[:, self.truth_index : self.truth_index + 1]
        self.pred_truth_labels = None

    def classify(self, unique_clusters, cluster_labels):
        "Classification start"
        total_keep, total_discard = 0, 0
        ground_truths = []
        pred_truth_labels = self.truth_labels
        for i in tqdm(unique_clusters):
            # count the number of keep and discard ground truth labels in each cluster
            cluster_points = self.truth_labels[cluster_labels == i]  # edit
            num_discard = 0
            for point in cluster_points:
                if point > float(0.5):
                    num_discard += 1
            num_keep = len(self.truth_labels[cluster_labels == i]) - num_discard
            print(
                "\nCluster", i, ": num_keep =", num_keep, "num_discard =", num_discard
            )
            total_keep += num_keep
            total_discard += num_discard

            # changing the clusters to keep and discard
            truth_val = float(1)  # 1 = discard
            if num_keep > num_discard:
                truth_val = float(0)  # 0 = keep
            ground_truths = np.append(ground_truths, truth_val)

            # set all points in cluster to majority ground truth of the cluster
            pred_truth_labels[cluster_labels == i] = truth_val

        assert len(ground_truths) == len(unique_clusters)
        print("\nFinal clusters:")
        print("ground_truths:", ground_truths, "\nunique_labels:", unique_clusters)
        print("total_discard:", total_discard, "total_keep:", total_keep)

        self.pred_truth_labels = pred_truth_labels
        return pred_truth_labels

    def get_attributes(self, arr, title):
        print("================", title, "\n:", arr)
        print("Length:", len(arr))
        print("Shape:", np.shape(arr))

    def visualise_classification(self, t):
        xyz = t[:, 0:3]
        predicted_ground_truth = self.predicted_labels.flatten()
        true_ground_truth = self.true_labels.flatten()
        print("predicted_labels:", predicted_ground_truth)
        print("true_labels:", true_ground_truth)
        # intensity1d = (t[:,3:4]).flatten()
        # pptk.viewer(xyz, intensity1d, predicted_ground_truth, true_ground_truth)
        pptk.viewer(xyz, predicted_ground_truth, true_ground_truth)
