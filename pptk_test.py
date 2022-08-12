import pptk 
import numpy as np

pcd = np.load("Visualise_Output\\raw_DBSCAN.npy")
points = pcd[:,0:3]
intensity = (pcd[:,3:4]).flatten()
truth = (pcd[:,4:5]).flatten()


l = len(pcd)
for i in range(l):
    if pcd[i,4:5][0] == float(1):
      print(pcd[i,4:5])

view = pptk.viewer(points, intensity, truth)

