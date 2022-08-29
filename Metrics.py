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
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import rand_score

import matplotlib.cm as cm


class Evaluation:
    def __init__(self, truth_labels):
        self.truth_labels = truth_labels
        
    def check_truth(self, y_pred, y_true):
        print("checking self.y_predict:")
        for i in y_pred:
            assert(not (i != float(0) and i != float(1)))
        print("checking self.y_true:")
        for i in y_true:
            assert(not (i != float(0) and i != float(1)))
    
    def evaluate_clusters(self, y_true, y_pred, cluster_labels, input_pcd, metric_choice = "all"):
        self.cluster_metrics = ClusterMetrics(y_true, y_pred, cluster_labels, input_pcd)
        scores = self.cluster_metrics.run_metric(metric_choice)
        return scores
    
    def evaluate_classification(self, y_true, y_pred, metric_choice = "all"):
        self.check_truth(y_pred, y_true)
        self.class_metrics = ClassificationMetrics(y_pred, y_true)
        scores = self.class_metrics.run_metric(metric_choice)
        return scores

class ClassificationMetrics:
    def __init__(self, y_pred, y_true) -> None:
        # Predicted Labels
        self.y_pred = y_pred
        # Ground Truth Labels
        self.y_true = y_true
        # Metrics
        self.classification_metrics = ['f1', 'jaccard', 'precision', 'recall', 'mean_abs', 'mean_sqr']
        
    
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
        elif metric_choice == "all":
            metric_vals = {}
            for metric in self.classification_metrics:
                metric_vals[metric] = self.run_metric(metric)
            return metric_vals

class ClusterMetrics:
    def __init__(self, y_true, y_pred, cluster_labels, input_pcd) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.cluster_labels = cluster_labels.flatten()
        self.input_pcd = input_pcd
        self.clustering_metrics = ['sill','db','rand']
        
    def run_metric(self, metric_choice):
        if metric_choice == "sill":
            return silhouette_score(self.input_pcd, self.cluster_labels)
        elif metric_choice == "db":
            return davies_bouldin_score(self.input_pcd, self.cluster_labels)
        elif metric_choice == "rand":
            return rand_score(self.y_true.flatten(), self.cluster_labels)
        elif metric_choice == "all":
            metric_vals = {}
            for metric in self.clustering_metrics:
                metric_vals[metric] = self.run_metric(metric)
                # print(metric_vals)
            return metric_vals