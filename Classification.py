import numpy as np
import pptk

class Classification:
    '''This class is reponsible for classifying a point cloud into 'keep' (0) and 'discard' (1) clusters'''
    def __init__(self):
      '''Initialise class parameters to None'''
      self.true_labels = None
      self.predicted_labels = None

    def get_ground_truth(self):
      '''Method to get the attribute for the true ground truth labels and the predicted ground truth labels
      returns:
        self.true_labels: an array containing the true ground truth labels of the point cloud
        self.predicted_labels: an array containing the predicted ground truth labels of the point cloud
      '''
      print("\n******************Get Ground Truth*******************")
      self.get_attributes(self.true_labels, "self.true_labels")
      self.get_attributes(self.predicted_labels, "self.predicted_labels")
      return self.true_labels, self.predicted_labels
          
    def classify(self, unique_labels, y_km, t, index):
      '''Method to label each produced cluster as the majority ground truth label in each cluster
      args:
        unique_labels: the cluster indexes
        y_km: the produced clusters
        t: the point cloud containing the actual ground truth at index
        index: the index of the actual ground truth in the array t
        '''
      print("\n******************Classification*******************")
      ground_truths = np.array([])
      self.true_labels = np.array(t[:, index:index+1])
      total_keep, total_discard = 0, 0

      # iterate through each cluster
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
      self.visualise_classification(t)
    
    def get_attributes(self, arr, title):
      '''Print attributes of a given array
      args: 
        arr: the array to print attributes of
        title: the title of the array to display'''
      print("================", title, "\n:", arr)
      print("Length:", len(arr), "\nShape:", np.shape(arr))

    def visualise_classification(self, t):
      '''Visualise the classified point cloud in pptk
      args: 
        t: the point cloud to display'''
      xyz = t[:,0:3]
      predicted_ground_truth = self.predicted_labels.flatten()
      true_ground_truth = self.true_labels.flatten()
      print("predicted_labels:", predicted_ground_truth, "\ntrue_labels:", true_ground_truth)
      pptk.viewer(xyz, predicted_ground_truth, true_ground_truth)