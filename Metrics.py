import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import *
import matplotlib.cm as cm
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

          plt.plot(list(results.keys()), list(results.values()))
          plt.xlabel("Number of clusters")
          plt.ylabel("Davies-Boulding Index")
          plt.show()
     
     def evaluate(self, metric_choice):
        if metric_choice == 0:
            # f1 score 
            score1 = f1_score(self.y_true, self.y_predict, pos_label=0)
            write_results_to_file("F1 Score (Binary):" + str(score1))
            score2 = f1_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("F1 Score (Macro):" + str(score2))
        elif metric_choice == 1:
            # IOU score
            score1 = jaccard_score(self.y_true, self.y_predict, pos_label=0)
            write_results_to_file("IOU Score (Binary):" + str(score1))
            score2 = jaccard_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("IOU Score (Macro):" + str(score2))
        elif metric_choice == 2:
            # precision
            score = precision_score(self.y_true, self.y_predict, pos_label=0)
            write_results_to_file("Precision (Binary):" + str(score1))
            score2 = precision_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("Precision (Macro):" + str(score2))
        elif metric_choice == 3:
            # recall
            score1 = recall_score(self.y_true, self.y_predict, pos_label=0)
            write_results_to_file("Recall (Binary):" + str(score1))
            score2 = recall_score(self.y_true, self.y_predict, average='macro')
            write_results_to_file("Recall (Macro):" + str(score2))
        elif metric_choice == 4:
            # mean absolute error 
            score1 = mean_absolute_error(self.y_true, self.y_predict)
            write_results_to_file("Mean Absolute Error:" + str(score1))
            score2 = ""
        elif metric_choice == 5:
            # mean squared error 
            score1 = mean_squared_error(self.y_true, self.y_predict)
            write_results_to_file("Mean Squared Error:" + str(score1))
            score2 = ""
        elif metric_choice == 6:
            # all metrics

            # binary results
            f = f1_score(self.y_true, self.y_predict, pos_label=0)
            j = jaccard_score(self.y_true, self.y_predict, pos_label=0)
            p = precision_score(self.y_true, self.y_predict, pos_label=0)
            r = recall_score(self.y_true, self.y_predict, pos_label=0)
            a = mean_absolute_error(self.y_true, self.y_predict)
            s = mean_squared_error(self.y_true, self.y_predict)
            print("F1 Score (Binary):", f, 
                  "\nIOU Score (Binary):", j, 
                  "\nPrecision (Binary):", p, 
                  "\nRecall (Binary):", r, 
                  "\nMean Absolute Error:", a,
                  "\nMean Squared Error:", s)

            write_results_to_file("F1 Score (Binary):" + str(f))
            write_results_to_file("IOU Score (Binary):" + str(j))
            write_results_to_file("Precision (Binary):" + str(p))
            write_results_to_file("Recall (Binary):" + str(r))
            write_results_to_file("Mean Absolute Error (Binary):" + str(s))
            write_results_to_file("Mean Squared Error (Binary):" + str(s))

            # macro results
            f2 = f1_score(self.y_true, self.y_predict, average='macro')
            j2 = jaccard_score(self.y_true, self.y_predict,  average='macro')
            p2 = precision_score(self.y_true, self.y_predict,  average='macro')
            r2 = recall_score(self.y_true, self.y_predict,  average='macro')
            print("F1 Score (Macro):", f2, 
                  "\nIOU Score (Macro):", j2, 
                  "\nPrecision (Macro):", p2, 
                  "\nRecall (Macro):", r2)
            score1 = ""
            score2 = ""

            write_results_to_file("F1 Score (Macro):" + str(f2))
            write_results_to_file("IOU Score (Macro):" + str(j2))
            write_results_to_file("Precision (Macro):" + str(p2))
            write_results_to_file("Recall (Macro):" + str(r2))
        return score1, score2

     def classification_metrics(self, actual_ground_truths, predicted_ground_truths):
        print("hi")
        write_results_to_file("*************Classification Metrics*************")
        # data
        self.y_true = actual_ground_truths 
        self.y_predict = predicted_ground_truths
        print("**********************INSIDE METRICS****************")
        print("---------------------self.y_true:", self.y_true)
        print("shape:", np.shape(self.y_true), "len:", len(self.y_true))
        print("---------------------self.y_predict:", self.y_predict)
        print("shape:", np.shape(self.y_predict), "len:", len(self.y_predict))

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
            score1, score2 = self.evaluate(int(userInput))
            print(score1)
            print(score2)

if __name__ == "__main__":
    t = Testing()
    t.classification_metrics()
