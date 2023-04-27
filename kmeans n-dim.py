import numpy as np
import csv
import multiprocessing
import visualisation


def getDataset(file):
    # data as numpy array
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    dimention = data.shape[1]
    return data, dimention


def saveResult(file, data, dimention):
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # write header like this: "x1, x2, x3, ..., centroid_x1, centroid_x2, centroid_x3, ..." depending on the dimention
        writer.writerow(["x" + str(i) for i in range(dimention)] + ["centroid_x" + str(i) for i in range(dimention)])
        csvfile.write("\n")
        for row in data:
            writer.writerow(row)
            # go to the next line
            csvfile.write("\n")


NB_CLUSTERS = 10
MAX_ITER = 100
NB_THREADS = 4

data, dimention = getDataset("./data/3d_data.csv")
centroids = data[np.random.choice(data.shape[0], NB_CLUSTERS, replace=False), :]
clusters = np.array([])
iteration = 0
convergence = False
old_mean_distance = float('inf')
mean_distances = float('inf')

while not convergence and iteration < MAX_ITER:
    # assign each point to the closest centroid
    # create a list of clusters
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)

    # compute the new centroids
    new_centroids = np.array([np.mean(data[clusters == i], axis=0) for i in range(NB_CLUSTERS)])

    # check convergence
    mean_distance = np.mean(np.linalg.norm(centroids - new_centroids, axis=1))
    if mean_distance == round(old_mean_distance, 5):
        convergence = True
    else:
        old_mean_distance = mean_distance
    centroids = new_centroids
    iteration += 1
    mean_distances = mean_distance

# format data for visualisation in a single array like this: [[x1, y1, centroid_x, centroid_y], [x2, y2, centroid_x, centroid_y], ...]
visualisation_data = np.concatenate((data, centroids[clusters]), axis=1)
visualisation.draw(visualisation_data)

# print mean distances between each point and its centroid
print(mean_distances)

# save the result in a csv file
saveResult("./data/3d_data_result.csv", visualisation_data, dimention)
