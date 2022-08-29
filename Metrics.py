import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import matplotlib.cm as cm


class Evaluation:
    def __init__(self, truth_labels):
        self.y_true = truth_labels[:,0:1]
        self.y_pred = truth_labels[:,1:2]
        self.class_metrics = ClassificationMetrics(self.y_pred, self.y_true)

    # def silhouette_kmeans(self):

    #     x = self.pcd

    #     K = range(2, 20)
    #     for k in K:
            
    #         clusterer = KMeans(n_clusters=k)  # for k-means and k-medoids

    #         cluster_labels = clusterer.fit_predict(x)

    #         silhouette_avg = silhouette_score(x, cluster_labels)
    #         print(
    #             "For n_clusters =",
    #             k,
    #             "The average silhouette_score is :",
    #             silhouette_avg,
    #         )
    #         sample_silhouette_values = silhouette_samples(x, cluster_labels)

    #         y_lower = 10
    #         for i in range(k):
    #             # Aggregate the silhouette scores for samples belonging to
    #             # cluster i, and sort them
    #             ith_cluster_silhouette_values = sample_silhouette_values[
    #                 cluster_labels == i
    #             ]

    #             ith_cluster_silhouette_values.sort()

    #             size_cluster_i = ith_cluster_silhouette_values.shape[0]
    #             y_upper = y_lower + size_cluster_i

    #             color = cm.nipy_spectral(float(i) / k)
    #             ax1.fill_betweenx(
    #                 np.arange(y_lower, y_upper),
    #                 0,
    #                 ith_cluster_silhouette_values,
    #                 facecolor=color,
    #                 edgecolor=color,
    #                 alpha=0.7,
    #             )

    #             ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    #             # Compute the new y_lower for next plot
    #             y_lower = y_upper + 10  # 10 for the 0 samples

    #         ax1.set_title("The silhouette plot for the various clusters.")
    #         ax1.set_xlabel("The silhouette coefficient values")
    #         ax1.set_ylabel("Cluster label")

    #         ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    #         ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #         ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    #         # 2nd Plot showing the actual clusters formed
    #         colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
    #         ax2.scatter(
    #             x[:, 0],
    #             x[:, 1],
    #             marker=".",
    #             s=30,
    #             lw=0,
    #             alpha=0.7,
    #             c=colors,
    #             edgecolor="k",
    #         )

    #         # Labeling the clusters
    #         centers = clusterer.cluster_centers_
    #         # Draw white circles at cluster centers
    #         ax2.scatter(
    #             centers[:, 0],
    #             centers[:, 1],
    #             marker="o",
    #             c="white",
    #             alpha=1,
    #             s=200,
    #             edgecolor="k",
    #         )

    #         for i, c in enumerate(centers):
    #             ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    #         ax2.set_title("The visualization of the clustered data.")
    #         ax2.set_xlabel("Feature space for the 1st feature")
    #         ax2.set_ylabel("Feature space for the 2nd feature")

    #         plt.suptitle(
    #             "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    #             % k,
    #             fontsize=14,
    #             fontweight="bold",
    #         )

    #     plt.show()

    # def db_index(self):
    #     x = self.pcd

    #     results = {}

    #     for i in range(2, 100):
    #         kmeans = KMeans(n_clusters=i, random_state=30)
    #         labels = kmeans.fit_predict(x)
    #         db_index = davies_bouldin_score(x, labels)
    #         results.update({i: db_index})
    #         print({i: db_index})

    #     plt.plot(list(results.keys()), list(results.values()))
    #     plt.xlabel("Number of clusters")
    #     plt.ylabel("Davies-Boulding Index")
    #     plt.show()
    
    def check_truth(self):
        print("checking self.y_predict:")
        for i in self.y_pred:
            assert(i != float(0) and i != float(1))
        print("checking self.y_true:")
        for i in self.y_true:
            assert(i != float(0) and i != float(1))
    
    def evaluate(self, metric_choice):
        self.check_truth()
        score = self.class_metrics.run_metric(metric_choice)
        print(score)
          
    # def evaluate(self, metric_choice):
    #     print("checking self.y_predict-----------------")
    #     for i in self.y_predict:
    #         if i != float(0) and i != float(1): print(i)
    #     print("checking self.y_true-----------------")
    #     for i in self.y_true:
    #         if i != float(0) and i != float(1): print(i)

    #     if metric_choice == 0:
    #         # f1 score 
    #         score = f1_score(self.y_true, self.y_predict, average='macro')
    #         write_results_to_file("F1 Score (Macro):" + str(score))
    #     elif metric_choice == 1:
    #         # IOU score
    #         score = jaccard_score(self.y_true, self.y_predict, average='macro')
    #         write_results_to_file("IOU Score (Macro):" + str(score))
    #     elif metric_choice == 2:
    #         # precision
    #         score = precision_score(self.y_true, self.y_predict, average='macro')
    #         write_results_to_file("Precision (Macro):" + str(score))
    #     elif metric_choice == 3:
    #         # recall
    #         score = recall_score(self.y_true, self.y_predict, average='macro')
    #         write_results_to_file("Recall (Macro):" + str(score))
    #     elif metric_choice == 4:
    #         # mean absolute error 
    #         score = mean_absolute_error(self.y_true, self.y_predict)
    #         write_results_to_file("Mean Absolute Error:" + str(score))
    #     elif metric_choice == 5:
    #         # mean squared error 
    #         score = mean_squared_error(self.y_true, self.y_predict)
    #         write_results_to_file("Mean Squared Error:" + str(score))
    #     elif metric_choice == 6:
    #         # all metrics
    #         f2 = f1_score(self.y_true, self.y_predict, average='macro')
    #         j2 = jaccard_score(self.y_true, self.y_predict,  average='macro')
    #         p2 = precision_score(self.y_true, self.y_predict,  average='macro')
    #         r2 = recall_score(self.y_true, self.y_predict,  average='macro')
    #         a = mean_absolute_error(self.y_true, self.y_predict)
    #         s = mean_squared_error(self.y_true, self.y_predict)
    #         print("F1 Score (Macro):", str(f2).replace('.', ','), 
    #               "\nIOU Score (Macro):", str(j2).replace('.', ','), 
    #               "\nPrecision (Macro):", str(p2).replace('.', ','), 
    #               "\nRecall (Macro):", str(r2).replace('.', ','),
    #               "\nMean Absolute Error:", str(a).replace('.', ','),
    #               "\nMean Squared Error:", str(s).replace('.', ','))
    #         score = ""

    #         write_results_to_file("F1 Score (Macro):" + str(f2).replace('.', ','))
    #         write_results_to_file("IOU Score (Macro):" + str(j2).replace('.', ','))
    #         write_results_to_file("Precision (Macro):" + str(p2).replace('.', ','))
    #         write_results_to_file("Recall (Macro):" + str(r2).replace('.', ','))
    #         write_results_to_file("Mean Absolute Error:" + str(a).replace('.', ','))
    #         write_results_to_file("Mean Squared Error:" + str(s).replace('.', ','))
    #     return score

class ClassificationMetrics:
    def __init__(self, y_pred, y_true) -> None:
        # Predicted Labels
        self.y_pred = y_pred
        # Ground Truth Labels
        self.y_true = y_true
    
    def run_metric(self, metric_choice):
        if metric_choice == "f1":
            return f1_score(self.y_true, self.y_pred)
        elif metric_choice == "jaccard":
            return jaccard_score(self.y_true, self.y_pred)
        elif metric_choice == "precision":
            return precision_score(self.y_true, self.y_pred)
        elif metric_choice == "recall":
            return recall_score(self.y_true, self.y_pred)
        elif metric_choice == "mean_abs":
            return mean_absolute_error(self.y_true, self.y_pred)
        elif metric_choice == "mean_sqr":
            return mean_squared_error(self.y_true, self.y_pred)
        # elif metric_choice == "all":
        #     # all metrics
        #     f2 = f1_score(self.y_true, self.y_predict, average='macro')
        #     j2 = jaccard_score(self.y_true, self.y_predict,  average='macro')
        #     p2 = precision_score(self.y_true, self.y_predict,  average='macro')
        #     r2 = recall_score(self.y_true, self.y_predict,  average='macro')
        #     ab = mean_absolute_error(self.y_true, self.y_predict)
        #     sq = mean_squared_error(self.y_true, self.y_predict)
        #     print("F1 Score (Macro):", str(f2).replace('.', ','), 
        #           "\nIOU Score (Macro):", str(j2).replace('.', ','), 
        #           "\nPrecision (Macro):", str(p2).replace('.', ','), 
        #           "\nRecall (Macro):", str(r2).replace('.', ','),
        #           "\nMean Absolute Error:", str(ab).replace('.', ','),
        #           "\nMean Squared Error:", str(sq).replace('.', ','))

class ClusterMetrics:
    def __init__(self) -> None:
        pass