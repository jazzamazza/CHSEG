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
        self.y_true = truth_labels[:,0:1]
        self.y_pred = truth_labels[:,1:2]
        self.class_metrics = ClassificationMetrics(self.y_pred, self.y_true)

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
    def __init__(self, y_true, y_pred, cluster_labels, input_pcd) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.cluster_labels = cluster_labels
        self.input_pcd = input_pcd
        
    def run_metric(self, metric_choice):
        if metric_choice == "sill":
            return silhouette_score(self.input_pcd, self.cluster_labels)
        if metric_choice == "db":
            return davies_bouldin_score(self.input_pcd, self.cluster_labels)
        if metric_choice == "rand":
            return rand_score(self.y_true, self.cluster_labels)