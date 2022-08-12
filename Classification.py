import numpy as np
from Outputting import write_results_to_file
import pptk

class Classification:
    def __init__(self):
      self.true_labels = None
      self.predicted_labels = None

    def get_ground_truth(self):
      print("\n******************Get Ground Truth*******************")
      self.get_attributes(self.true_labels, "self.true_labels")
      self.get_attributes(self.predicted_labels, "self.predicted_labels")
      return self.true_labels, self.predicted_labels
          
    # for raw: index = 4 --> x, y, z, intensity, ground_truth
    # for cloud compare: index = 4 --> x, y, z, intensity, ground_truth, ...
    # for pointnet: index = 3 --> x, y, z, ground_truth, ...
    def classify(self, unique_labels, y_km, t, index, file_path, file_name):
      print("\n******************Classification*******************")
      
      self.true_labels = np.array([])
      ground_truths = np.array([])
      print("t[0]", t[0])
      first_label = True

      total_keep, total_discard = 0, 0
      for i in unique_labels:
            # count the number of keep and discard ground truth labels in each cluster
            num_discard = sum([1 for point in t[y_km == i] if (point[index] > float(0))])
            num_keep = len(t[y_km == i]) - num_discard
            print("num_keep:", num_keep)
            print("num_discard:", num_discard)
            total_keep += num_keep
            total_discard += num_discard

            # normalise true labels to values of either 1 or 0:
            norm_true = [[float(1)] if (point[index] > float(0)) else [float(0)] for point in t[y_km == i] ]
            if first_label:
                  self.true_labels = norm_true
                  first_label = False
            else:
                  self.true_labels = np.vstack((self.true_labels, norm_true))

            # changing the clusters to keep and discard
            truth_val = float(1) # 0 = keep
            if num_keep > num_discard: truth_val = float(0) # 1 = discard
            
            ground_truths = np.append(ground_truths, truth_val)

            # set all points in cluster to majority ground truth of the cluster
            t[y_km == i, index:index+1] = truth_val
            
      print("ground_truth:", ground_truths)
      print("unique_labels:", unique_labels)
      assert len(ground_truths) == len(unique_labels)
            
      print("t shape", np.shape(t))
      print("t[0]", t[0])
      print("t", t)

      print("total_discard:", total_discard)
      print("total_keep:", total_keep)

      class_file_name = file_path + file_name + ".npy"
      np.save(class_file_name, t)
      write_results_to_file("-------Classified PCD filename: " + class_file_name + "-------")
      self.predicted_labels = t[:,index:index+1] # predicted_ground_truths

      self.visualise_classification(t)
    
    def get_attributes(self, arr, title):
      print("================", title, "\n:", arr)
      print("Length:", len(arr))
      print("Shape:", np.shape(arr))

    def visualise_classification(self, t):
      xyz = t[:,0:3]
      intensity1d = (t[:,3:4]).flatten()
      predicted_ground_truth = self.predicted_labels.flatten()
      true_ground_truth = self.true_labels.flatten()
      print("predicted_labels:", predicted_ground_truth)
      print("true_labels:", true_ground_truth)
      pptk.viewer(xyz, intensity1d, predicted_ground_truth, true_ground_truth)
      print("pptk loaded")