
# metrics
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import rand_score

class Evaluation:
    def __init__(self, truth_labels):
        """Evaluation Class

        Args:
            truth_labels (ndarray): ground truth label array (n, 1)
        """
        self.truth_labels = truth_labels

    def check_truth(self, y_pred, y_true):
        """Check binary labelling of ground truth labels 

        Args:
            y_pred (ndarray): Predicted truth labels
            y_true (ndarray): Ground truth labels
        """
        # check all values 1 or 0
        print("checking self.y_predict:")
        for i in y_pred:
            assert not (i != float(0) and i != float(1))
        print("All good!")
        # check all values 1 or 0
        print("checking self.y_true:")
        for i in y_true:
            assert not (i != float(0) and i != float(1))
        print("All good!")

    def evaluate_clusters(
        self,
        y_true,
        y_pred,
        cluster_labels,
        input_pcd,
        clustering_metrics=["sill", "db", "rand"],
        metric_choice="all",
    ):
        """Evaluate clustering using cluster metrics.

        Args:
            y_pred (ndarray): Predicted truth labels.
            y_true (ndarray): Ground truth labels.
            cluster_labels (ndarray): Per point cluster labels (n, n clusters).
            input_pcd (ndarray): Input point cloud data to clustering.
            clustering_metrics (list, optional): List of clustering metrics to evaluate. Defaults to ["sill", "db", "rand"].
            metric_choice (str, optional): Evaluation(s) e.g. "db". Defaults to "all".

        Returns:
            list: List of scores for each or one metric.
        """
        # create metric object
        self.cluster_metrics = ClusterMetrics(y_true, y_pred, cluster_labels, input_pcd)
        # set up run
        self.cluster_metrics.set_metrics(clustering_metrics)
        # evaluate clustering
        scores = self.cluster_metrics.run_metric(metric_choice)
        return scores

    def evaluate_classification(
        self,
        y_true,
        y_pred,
        classification_metrics=[
            "f1",
            "jaccard",
            "precision",
            "recall",
            "mean_abs",
            "mean_sqr",
        ],
        metric_choice="all",
    ):
        """Evaluate classification using classification metrics.

        Args:
            y_pred (ndarray): Predicted truth labels.
            y_true (ndarray): Ground truth labels.
            classification_metrics (list, optional): List of classification metrics to evaluate. Defaults to [ "f1", "jaccard", "precision", "recall", "mean_abs", "mean_sqr", ].
            metric_choice (str, optional): Evaluation(s) e.g. "f1". Defaults to "all".

        Returns:
            list: List of scores for each or one metric.
        """
        # very binary classification
        self.check_truth(y_pred, y_true)
        # create metric object
        self.class_metrics = ClassificationMetrics(y_pred, y_true)
        # set up
        self.class_metrics.set_metrics(classification_metrics)
        # evaluate
        scores = self.class_metrics.run_metric(metric_choice)
        return scores


class ClassificationMetrics:
    def __init__(self, y_pred, y_true) -> None:
        """ClassificationMetrics class. Provides evaulation of binary classification.

        Args:
            y_pred (ndarray): Predicted truth labels.
            y_true (ndarray): Ground truth labels.
        """
        # Predicted Labels
        self.y_pred = y_pred
        # Ground Truth Labels
        self.y_true = y_true
        # Metrics (default)
        self.classification_metrics = [
            "f1",
            "jaccard",
            "precision",
            "recall",
            "mean_abs",
            "mean_sqr",
        ]

    def set_metrics(self, metrics):
        # update metrics
        self.classification_metrics = metrics

    def run_metric(self, metric_choice):
        """Evaluate one or all clustering metrics.

        Args:
            metric_choice (str): Metric chosen to be evaluated

        Returns:
            list: List of scores for each or one metric.
        """
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
        """ClusterMetrics class. Evalute clustering.

        Args:
            y_pred (ndarray): Predicted truth labels.
            y_true (ndarray): Ground truth labels.
            cluster_labels (ndarray): Per point cluster labels (n, n clusters).
            input_pcd (ndarray): Input point cloud data to clustering.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.cluster_labels = cluster_labels.flatten()
        self.input_pcd = input_pcd
        # default metrics
        self.clustering_metrics = ["sill", "db", "rand"]

    def set_metrics(self, metrics):
        # update metrics
        self.clustering_metrics = metrics

    def run_metric(self, metric_choice):
        """Evaluate one or all clustering metrics.

        Args:
            metric_choice (str): Metric chosen to be evaluated

        Returns:
            list: List of scores for each or one metric.
        """
        if metric_choice == "sill":
            print("run silhoutte")
            return silhouette_score(self.input_pcd, self.cluster_labels)
        elif metric_choice == "db":
            print("run db")
            return davies_bouldin_score(self.input_pcd, self.cluster_labels)
        elif metric_choice == "rand":
            return rand_score(self.y_true.flatten(), self.cluster_labels)
        elif metric_choice == "all":
            metric_vals = {}
            for metric in self.clustering_metrics:
                metric_vals[metric] = self.run_metric(metric)
                # print(metric_vals)
            return metric_vals
