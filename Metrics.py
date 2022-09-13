from sklearn.cluster import KMeans
from sklearn.metrics import *
from Outputting import *

class Testing:
    '''This class is responsible for evaluating the clustering and classification results'''
    def __init__(self, pointCloud, out):
        '''Initialise class variables
        Args:
            pointCloud: the point cloud dataset to evaluate
            out: the Outputting class to write results to'''
        self.pcd = pointCloud
        self.out = out

    def silhouette_kmeans(self, upperBound, lowerBound):
        '''Calculate the average silhouette score for up to upperBound values of k'''
        for k in range(lowerBound, upperBound):             
            cluster_labels = KMeans(n_clusters= k).fit_predict(self.pcd)
            print("For", k, "clusters, the average silhouette score is:", silhouette_score(self.pcd, cluster_labels))

    def db_index(self, upperBound, lowerBound):
        '''Calculate the davies bouldin score for up to upperBound values of k'''
        for k in range(lowerBound, upperBound):
            labels = KMeans(n_clusters=k, random_state=30).fit_predict(self.pcd)
            print("For", k, "clusters, the davies bouldin score is:", davies_bouldin_score(self.pcd, labels))
     
    def evaluate(self, metric_choice):
        '''Evaluate the classification result using classification evaluation metrics
        Args: 
            metric_choice: an integer corresponding to a classification evaluation metric
        Returns:
            score: the evaluation metric score
            metric_name: the name of the metric evaluated
        '''
        if metric_choice == 0:
            # f1 score 
            score = f1_score(self.y_true, self.y_predict, average='macro')
            metric_name = "F1 Score (Macro):"
        elif metric_choice == 1:
            # IOU score
            score = jaccard_score(self.y_true, self.y_predict, average='macro')
            metric_name = "IOU Score (Macro):"
        elif metric_choice == 2:
            # precision
            score = precision_score(self.y_true, self.y_predict, average='macro')
            metric_name = "Precision (Macro):"
        elif metric_choice == 3:
            # recall
            score = recall_score(self.y_true, self.y_predict, average='macro')
            metric_name = "Recall (Macro):"
        elif metric_choice == 4:
            # mean absolute error 
            score = mean_absolute_error(self.y_true, self.y_predict)
            metric_name = "Mean Absolute Error:"
        elif metric_choice == 5:
            # mean squared error 
            score = mean_squared_error(self.y_true, self.y_predict)
            metric_name = "Mean Squared Error:"
        elif metric_choice == 6:
            # all metrics
            f2 = f1_score(self.y_true, self.y_predict, average='macro')
            j2 = jaccard_score(self.y_true, self.y_predict,  average='macro')
            p2 = precision_score(self.y_true, self.y_predict,  average='macro')
            r2 = recall_score(self.y_true, self.y_predict,  average='macro')
            a = mean_absolute_error(self.y_true, self.y_predict)
            s = mean_squared_error(self.y_true, self.y_predict)
            score, metric_name = None, None

            self.out.write_results(["F1 Score (Macro):" + str(f2).replace('.', ','),
            "IOU Score (Macro):" + str(j2).replace('.', ','),
            "Precision (Macro):" + str(p2).replace('.', ','),
            "Recall (Macro):" + str(r2).replace('.', ','),
            "Mean Absolute Error:" + str(a).replace('.', ','),
            "Mean Squared Error:" + str(s).replace('.', ',')])

            print("All results written to file")
        return score, metric_name

    def classification_metrics(self, actual_ground_truths, predicted_ground_truths):
        '''Compares the actual ground truth labels with the predicted ground truth labels'''
        self.out.write_results_to_file("*************Classification Metrics*************")
        
        self.y_true = actual_ground_truths 
        self.y_predict = predicted_ground_truths

        userInput = ""
        while (userInput != "q"):
            userInput = input("\nChoose Classification Metric to Evaluate with:"+
                                    "\n 0 : F1 Score" +
                                    "\n 1 : Intersection Over Union Score"+
                                    "\n 2 : Precision"+
                                    "\n 3 : Recall"+
                                    "\n 4 : Mean Absolute Error"+
                                    "\n 5 : Mean Squared Error"+
                                    "\n 6 : All of the Above"+
                                    "\n q : Quit\n")
            if (userInput == "q"): break
            score, metric_name = self.evaluate(int(userInput))
            if score and metric_name:
                self.out.write_results_to_file(metric_name + str(score))
                print(score)

if __name__ == "__main__":
    t = Testing()
    t.classification_metrics()
