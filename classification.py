import numpy as np
import pptk

# classification class 
class Classification:
    def __init__(self):
      # actual ground truth labels 
      self.true_labels = None
      # predicted ground truth labels 
      self.predicted_labels = None


    def get_ground_truth(self):
      print("\n****************** Get Ground Truth *******************")
      self.get_attributes(self.true_labels, "self.true_labels")
      self.get_attributes(self.predicted_labels, "self.predicted_labels")
      return self.true_labels, self.predicted_labels

    #classification task: obtaining the predicted ground truth labels 
    def classify(self, unique_labels, y_km, t, index, file_path, file_name):
      print("\n****************** Classification *******************")
      
      # set ground truths to an empty array 
      ground_truths = np.array([])
      # ceil the ground truth labels so the labels are 0 and 1 
      self.true_labels = np.array(np.ceil(t[:, index:index+1]))
      total_keep, total_discard = 0, 0              # initialise the keep and discard to zero 
      for i in unique_labels:
            # count the number of keep and discard ground truth labels in each cluster
            num_discard = sum([1 for point in t[y_km == i] if (point[index] > float(0.5))])
            num_keep = len(t[y_km == i]) - num_discard
            print("num_keep:", num_keep, "num_discard:", num_discard)
            total_keep += num_keep
            total_discard += num_discard

            # changing the clusters to keep and discard
            truth_val = float(1) # 0 = keep
            if num_keep > num_discard: truth_val = float(0) # 1 = discard
            ground_truths = np.append(ground_truths, truth_val)

            # set all points in cluster to majority ground truth of the cluster
            t[y_km == i, index:index+1] = truth_val
            
      assert len(ground_truths) == len(unique_labels)
      print("ground_truth:", ground_truths, "\nunique_labels:", unique_labels)
      print("total_discard:", total_discard, "total_keep:", total_keep)

      
      self.predicted_labels = t[:,index:index+1] # predicted_ground_truths
      self.visualise_classification(t)        # visualise the classified results 
    
    # get the attributes of the point cloud: length and shape 
    def get_attributes(self, arr, title):
      print("================", title, "\n:", arr)
      print("Length:", len(arr))
      print("Shape:", np.shape(arr))

    #visualization method for the classification results: actual ground truth vs predicted
    def visualise_classification(self, t):
      xyz = t[:,0:3]
      predicted_ground_truth = self.predicted_labels.flatten()
      true_ground_truth = self.true_labels.flatten()
      print("predicted_labels:", predicted_ground_truth)
      print("true_labels:", true_ground_truth)
      pptk.viewer(xyz, predicted_ground_truth, true_ground_truth)

