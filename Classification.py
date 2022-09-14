import numpy as np
from tqdm import tqdm
class Classification:
    def __init__(self, ground_truth_labels):
        """Classification class runs Binary Classification

        Args:
            ground_truth_labels (ndarray): numpy array of ground truth labels
        """
        #initially set truth and pred truth to same values
        self.truth_labels = ground_truth_labels
        self.pred_truth_labels = np.copy(self.truth_labels)
        
        #set when classify is called
        self.cluster_labels = None
        self.unique_clusters = None

    def classify(self, unique_clusters, cluster_labels):
        """Run Binary Classification on Dataset

        Args:
            unique_clusters (ndarray): list of unique cluster names i.e. [1,2,3] if 
            cluster_labels (ndarray): numpy array of cluster labels per point
        """
        print("*** Binary classification start ***")
        self.unique_clusters = unique_clusters
        self.cluster_labels = cluster_labels
        #init counters
        total_keep, total_discard = 0, 0
        #holds predicted ground truths
        ground_truths = []
        for i in tqdm(unique_clusters):
            # count the number of keep and discard ground truth labels in each cluster
            cluster_points = self.truth_labels[cluster_labels == i]
            num_discard = 0
            for point in cluster_points:
                #if point is labled discard (>0.5)
                if point > float(0.5):
                    num_discard += 1
            num_keep = len(self.truth_labels[cluster_labels == i]) - num_discard
            total_keep += num_keep
            total_discard += num_discard
            
            # changing the clusters to keep and discard
            truth_val = float(1)  # 1 = discard
            if num_keep > num_discard:
                truth_val = float(0)  # 0 = keep
            # list of truth label per cluster
            ground_truths = np.append(ground_truths, truth_val)
            # set all points in cluster to majority ground truth of the cluster
            self.pred_truth_labels[cluster_labels == i] = truth_val
        # make sure number of clusters in pred ground truth and cluster labels is the same
        assert len(ground_truths) == len(unique_clusters)
        # make sure predicted truth labels and ground truth labels are no longer the same
        assert (not np.array_equal(self.truth_labels, self.pred_truth_labels))
        print("\nFinal clusters:")
        print("ground_truths_len:",len(ground_truths),
            "\nunique_labels_len:",len(unique_clusters))
        n_points = total_discard+total_keep
        print("n points:",n_points,"total_discard:", total_discard, "total_keep:", total_keep)