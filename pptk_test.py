import pptk 
import numpy as np

pcd = np.load("Visualise_Output\\raw_meanshift.npy")
points = pcd[:,0:3]
intensity = (pcd[:,3:4]).flatten()
truth = (pcd[:,4:5]).flatten()

view = pptk.viewer(points, intensity, truth)

