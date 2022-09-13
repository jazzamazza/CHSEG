from sklearn.cluster import KMeans
from sklearn.metrics import *
from Outputting import write_results_to_file

class Testing:
     def __init__(self, pointCloud):
          self.pcd = pointCloud

     def silhouette_kmeans(self):
          x = self.pcd
          K = range(2, 20)
          for k in K:
               clusterer = KMeans(n_clusters= k)              
               cluster_labels = clusterer.fit_predict(x)
               silhouette_avg = silhouette_score(x, cluster_labels)
               print(
                    "For n_clusters =", k,
                    "The average silhouette_score is :",
                         silhouette_avg,
               )

     def db_index(self):
          x = self.pcd
          results = {}
          for i in range(2,100):
               kmeans = KMeans(n_clusters=i, random_state=30)
               labels = kmeans.fit_predict(x)
               db_index = davies_bouldin_score(x, labels)
               results.update({i: db_index})
               print({i: db_index})
     
     def evaluate(self, metric_choice):
        if metric_choice == 0:
            # f1 score 
            score = f1_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("F1 Score (Macro):" + str(score))
        elif metric_choice == 1:
            # IOU score
            score = jaccard_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("IOU Score (Macro):" + str(score))
        elif metric_choice == 2:
            # precision
            score = precision_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("Precision (Macro):" + str(score))
        elif metric_choice == 3:
            # recall
            score = recall_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("Recall (Macro):" + str(score))
        elif metric_choice == 4:
            # mean absolute error 
            score = mean_absolute_error(self.y_true, self.y_predict)
            write_results_to_file("Mean Absolute Error:" + str(score))
        elif metric_choice == 5:
            # mean squared error 
            score = mean_squared_error(self.y_true, self.y_predict)
            write_results_to_file("Mean Squared Error:" + str(score))
        elif metric_choice == 6:
            # all metrics
            f2 = f1_score(self.y_true, self.y_predict, average='macro')
            j2 = jaccard_score(self.y_true, self.y_predict,  average='macro')
            p2 = precision_score(self.y_true, self.y_predict,  average='macro')
            r2 = recall_score(self.y_true, self.y_predict,  average='macro')
            a = mean_absolute_error(self.y_true, self.y_predict)
            s = mean_squared_error(self.y_true, self.y_predict)
            print("F1 Score (Macro):", str(f2).replace('.', ','), 
                  "\nIOU Score (Macro):", str(j2).replace('.', ','), 
                  "\nPrecision (Macro):", str(p2).replace('.', ','), 
                  "\nRecall (Macro):", str(r2).replace('.', ','),
                  "\nMean Absolute Error:", str(a).replace('.', ','),
                  "\nMean Squared Error:", str(s).replace('.', ','))
            score = ""

            write_results_to_file("F1 Score (Macro):" + str(f2).replace('.', ','))
            write_results_to_file("IOU Score (Macro):" + str(j2).replace('.', ','))
            write_results_to_file("Precision (Macro):" + str(p2).replace('.', ','))
            write_results_to_file("Recall (Macro):" + str(r2).replace('.', ','))
            write_results_to_file("Mean Absolute Error:" + str(a).replace('.', ','))
            write_results_to_file("Mean Squared Error:" + str(s).replace('.', ','))
        return score

     def classification_metrics(self, actual_ground_truths, predicted_ground_truths):
        write_results_to_file("*************Classification Metrics*************")
        # data
        self.y_true = actual_ground_truths 
        self.y_predict = predicted_ground_truths

        userInput = "", ""
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
            score = self.evaluate(int(userInput))
            print(score)

if __name__ == "__main__":
    t = Testing()
    t.classification_metrics()
