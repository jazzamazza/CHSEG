import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data_folder = "./data/"
data_file = "KME_planes.xyz"

x,y,z,illuminance,reflectance,intensity,nb_of_returns = np.loadtxt(data_folder+data_file, skiprows=1, delimiter=';', unpack=True)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.scatter(x, z, c=intensity, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("First view")
plt.xlabel('X-axis ')
plt.ylabel('Z-axis ')

# If you look within the lines, I use the intensity 
# field as the coloring element for our plot. 
# I can do this because it is already normalized at an
# [0,1] interval. The s stands for size and permits us 
# to give a size to our points.

plt.subplot(1, 2, 2) # index 2
plt.scatter(y, z, c=intensity, s=0.05)
plt.axhline(y=np.mean(z), color='r', linestyle='-')
plt.title("Second view")
plt.xlabel('Y-axis ')
plt.ylabel('Z-axis ')

plt.show()

pcd=np.column_stack((x,y,z))
mask=z>np.mean(z)
spatial_query=pcd[z>np.mean(z)]

print(pcd.shape==spatial_query.shape)

#plotting the results 3D
ax = plt.axes(projection='3d')
ax.scatter(x[mask], y[mask], z[mask], c = intensity[mask], s=0.1)
plt.show()

#plotting the results 2D
plt.scatter(x[mask], y[mask], c=intensity[mask], s=0.1)
plt.show()

X=np.column_stack((x[mask], y[mask]))

kmeans = KMeans(n_clusters=4).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

X=np.column_stack((x[mask], y[mask], z[mask]))
wcss = [] 
for i in range(1, 20):
 kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
 kmeans.fit(X)
 wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

data_folder="./data/"
dataset="KME_cars.xyz"
x,y,z,r,g,b = np.loadtxt(data_folder+dataset,skiprows=1, delimiter=';', unpack=True)
X=np.column_stack((x,y,z))
kmeans = KMeans(n_clusters=3).fit(X)

#analysis on dbscan
clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
plt.scatter(x, y, c=clustering.labels_, s=20)
plt.show()

X=np.column_stack((x[mask], y[mask], z[mask], illuminance[mask], nb_of_returns[mask], intensity[mask]))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

X=np.column_stack((z[mask] ,z[mask], intensity[mask]))
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
plt.scatter(x[mask], y[mask], c=kmeans.labels_, s=0.1)
plt.show()

result_folder="./data/results/"
np.savetxt(result_folder+dataset.split(".")[0]+"_result.xyz", np.column_stack((x[mask], y[mask], z[mask],kmeans.labels_)), fmt='%1.4f', delimiter=';')