import numpy as np
# import pptk

class Classification:
    def __init__(self):
          self.true_labels = None
          self.predicted_labels = None

    def get_ground_truth(self):
          return self.true_labels, self.predicted_labels
          
    # for raw: index = 4
    # for cloud compare: index = 
    # for pointnet: index =
    def classify(self, unique_labels, y_km, t, index, file_path, file_name):
          num_keep, num_discard = 0, 0
          self.true_labels = t[:,index:index+1] # ground truth before processing

          ground_truths = np.array([])
          print("t[0]", t[0])
          print("ground_truth size:", ground_truths.size)
          for i in unique_labels:
              num_keep, num_discard = 0, 0
              for point in t[y_km == i]:
                if (point[index] >= float(0.5)): num_discard += 1
                else: num_keep += 1
              print("num_keep:", num_keep)
              print("num_discard:", num_discard)
              if num_keep > num_discard: 
                ground_truths = np.append(ground_truths, 1) #changing the clusters to keep and discard
              else: 
                ground_truths = np.append(ground_truths, 0)

          print("ground_truth:", ground_truths)

          #sets cluster to majority ground truth       
          for i in range(0, len(ground_truths)):   #i is each cluster
            if ground_truths[i] == float(1): # if cluster == keep
              for point in t[y_km == i]: # set ground truth of each point to keep
                # print("point", point)
                t[y_km == i, index:index+1] = float(1)
            else:
              for point in t[y_km == i]:
                t[y_km == i, index:index+1] = float(0)
                
          print("t shape", np.shape(t))
          print("truth", t[0])
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
      