import numpy as np
# import pptk

class Classification:
    def __init__(self):
          self.true_labels = None
          self.predicted_labels = None

    def get_ground_truth(self):
          return self.true_labels, self.predicted_labels
          
    # for raw: index = 4 --> x, y, z, intensity, ground_truth
    # for cloud compare: index = 4 --> x, y, z, intensity, ground_truth, ...
    # for pointnet: index = 3 --> x, y, z, ground_truth, ...
    def classify(self, unique_labels, y_km, t, index, file_path, file_name):
          print("\n******************Classification*******************")

          self.true_labels = t[:,index:index+1] # ground truth before processing
          ground_truths = np.array([])
          print("t[0]", t[0])
          
          for i in unique_labels:
              # count the number of keep and discard ground truth labels in each cluster
              num_discard = sum([1 for point in t[y_km == i] if (point[index] >= float(0.5))])
              num_keep = len(t[y_km == i]) - num_discard
              print("num_keep:", num_keep)
              print("num_discard:", num_discard)

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

          np.save(file_path + file_name + ".npy", t)
          self.predicted_labels = t[:,index:index+1] # predicted_ground_truths
    
    # def visualise_classification(self, pcd):
    #       print("self.predicted_labels.flatten()", self.predicted_labels.flatten())
    #       xyz = pcd[:,0:3]
    #       intensity1d = (pcd[:,3:4]).flatten()
    #       view = pptk.viewer(xyz, intensity1d)
    #       # view = pptk.viewer(xyz, intensity1d, t[:,index:index+1].flatten()) 
    #       print("pptk loaded")
      