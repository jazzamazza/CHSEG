from sklearn.metrics import *

class classification_metrics:
    def __init__(self, actual_ground_truths=[0,1,1,0,0,1], predicted_ground_truths=[0,0,1,1,0,1]):
        # data
        self.y_true = actual_ground_truths 
        self.y_predict = predicted_ground_truths
    
    def evaluate(self, metric_choice):
        if metric_choice == 0:
            # f1 score 
            score = f1_score(self.y_true, self.y_predict)
        elif metric_choice == 1:
            # IOU score
            score = jaccard_score(self.y_true, self.y_predict)
        elif metric_choice == 2:
            # precision
            score = precision_score(self.y_true, self.y_predict)
        elif metric_choice == 3:
            # recall
            score = recall_score(self.y_true, self.y_predict)
        elif metric_choice == 4:
            # error rate 
            score = max_error(self.y_true, self.y_predict)
        return score

if __name__ == "__main__":
    c = classification_metrics()
    score = ""
    userInput = ""
    while (userInput != "q"):
        userInput = input("\nChoose Classification Metric to Evaluate with:"+
                                "\n 0 : F1 Score" +
                                "\n 1 : Intersection Over Union Score"+
                                "\n 2 : Precision"+
                                "\n 3 : Recall"+
                                "\n 4 : Error Rate\n")
        score = c.evaluate(int(userInput))
        print(score)


