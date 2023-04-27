import csv
import random
import visualisation
import numpy as np
import time

def getDataset(file):
    file = open(file, 'r')
    reader = csv.reader(file)
    next(reader)
    data = list(reader)
    file.close()
    for i in range(len(data)):
        data[i] = [float(data[i][0]), float(data[i][1])]
    return data

def save(data):
    file = open("./data/data_out.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(["x", "y"])
    for i in range(len(data)):
        writer.writerow(data[i])
    file.close()

def distance(x1, y1, x2, y2):
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

def find_closest_centroid(x,y, centroids):
    closest_centroid = 0
    closest_distance = ((centroids[0][0] - x)**2 + (centroids[0][1] - y)**2)**0.5
    for i in range(1, len(centroids)):
        new_distance = distance(x, y, centroids[i][0], centroids[i][1])
        if new_distance < closest_distance:
            closest_distance = new_distance
            closest_centroid = i
    return closest_centroid

def getbarycenter(cluster):
    #if cluster is empty return error
    if len(cluster) == 0:
        return None
    x = 0
    y = 0
    for point in cluster:
        x += point[0]
        y += point[1]
    return [x/len(cluster), y/len(cluster)]

def get_mean_distance(centroids, clusters):
    if len(centroids) != len(clusters) or len(centroids) == 0 or len(clusters) == 0 :
        return None
    mean_distance = 0
    mean_distance_c = 0
    for i in range(len(centroids)):
        for point in clusters[i]:
            mean_distance_c += distance(point[0], point[1], centroids[i][0], centroids[i][1])
            #true mean distance is the mean of the mean distance of each cluster
        mean_distance_c /= len(clusters[i])
        mean_distance += mean_distance_c
    mean_distance /= len(centroids)
    return mean_distance

def kmeans(data, clusters_number):
    clusters = []
    for i in range(clusters_number):
        clusters.append([])
    centroids = []
    for i in range(clusters_number):
        centroids.append(data[random.randint(0, len(data)-1)])
    
    mean_distance = 0
    old_mean_distance = float('inf')
    cpt = 0
    while mean_distance != old_mean_distance and cpt < 10:
        cpt += 1
        for i in range(len(clusters)):
            clusters[i].clear()
        old_mean_distance = mean_distance
        for point in data:
            closeset_centroid = find_closest_centroid(point[0], point[1], centroids)
            clusters[closeset_centroid].append(point)
        new_centroids = []
        for cluster in clusters:
            barycentre = getbarycenter(cluster)
            if barycentre == None:
                return None, None
            new_centroids.append(barycentre)
        centroids = new_centroids.copy()
        new_centroids.clear()
        mean_distance = get_mean_distance(centroids, clusters)
    #put data in the right format for visualisation data[i]=[x, y, cx, cy]
    for i in range(len(data)):
        data[i].append(centroids[find_closest_centroid(data[i][0], data[i][1], centroids)][0])
        data[i].append(centroids[find_closest_centroid(data[i][0], data[i][1], centroids)][1])
    print (cpt)
    return data,mean_distance

start_time = time.time()
best_mean_distance = 100000000000
best_data = []
data_in = []
for i in range(1):
    data = getDataset('./data/2d_data.csv').copy()
    data_in.clear()
    data_in = data.copy()
    data2, mean_distance = kmeans(data_in, 10)
    if mean_distance < best_mean_distance:
        best_mean_distance = mean_distance
        best_data.clear()
        best_data = data2.copy()
        print (best_mean_distance)
print (best_mean_distance)
save(best_data)
exec_time = time.time() - start_time
dimention = 2
dataset_size = len(best_data)
performance_score = dimention* dataset_size / (exec_time*best_mean_distance)
print (performance_score)
visualisation.draw(best_data)