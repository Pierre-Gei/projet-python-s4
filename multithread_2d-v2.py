import numpy as np
import csv
import multiprocessing
import visualisation

def getDataset(file):
    #data as numpy array
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    dimention = data.shape[1]
    return data,dimention


NB_CLUSTERS = 10
MAX_ITER = 100
NB_THREADS = 4


data,dimention = getDataset("./data/2d_data.csv")
centroids = data[np.random.choice(data.shape[0], NB_CLUSTERS, replace=False), :]
clusters = np.array([])
iteration = 0
convergence = False
old_mean_distance = float('inf')
while not convergence and iteration < MAX_ITER:
    #assign each point to the closest centroid
    #create a list of clusters
    clusters = [[] for i in range(NB_CLUSTERS)]
    for point in data:
        closest_centroid = 0
        closest_distance = np.linalg.norm(point - centroids[0])
        for i in range(1, NB_CLUSTERS):
            new_distance = np.linalg.norm(point - centroids[i])
            if new_distance < closest_distance:
                closest_distance = new_distance
                closest_centroid = i
        clusters[closest_centroid].append(point)
    #compute the new centroids
    new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    #check convergence
    mean_distance = np.mean([np.linalg.norm(centroids[i] - new_centroids[i]) for i in range(NB_CLUSTERS)])
    if mean_distance == round(old_mean_distance, 5):
        convergence = True
    else:
        old_mean_distance = mean_distance
    centroids = new_centroids
    iteration += 1

#format data for visualisation in a sigle array like this: [[x1, y1, centroid_x, centroid_y], [x2, y2, centroid_x, centroid_y], ...]
visualisation_data = np.array([[point[0], point[1], centroids[i][0], centroids[i][1]] for i in range(len(clusters)) for point in clusters[i]])
visualisation.draw(visualisation_data)