import numpy as np


class Classification:
    def classify(self, n_clusters, cluster_labels, truth_labels):
        total_keep, total_discard = 0, 0
        for i in n_clusters:
            # count the number of keep and discard ground truth labels in each cluster
            cluster_points = None  # edit
            num_discard = sum(
                [
                    1
                    for point in truth_labels[cluster_labels == i]
                    if (point[index] > float(0.5))
                ]
            )
            num_keep = len(t[y_km == i]) - num_discard
            print("num_keep:", num_keep, "num_discard:", num_discard)
            total_keep += num_keep
            total_discard += num_discard

            # changing the clusters to keep and discard
            truth_val = float(1)  # 0 = keep
            if num_keep > num_discard:
                truth_val = float(0)  # 1 = discard
            ground_truths = np.append(ground_truths, truth_val)

            # set all points in cluster to majority ground truth of the cluster
            t[y_km == i, index : index + 1] = truth_val
