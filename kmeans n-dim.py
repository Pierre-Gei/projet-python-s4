import numpy as np
import csv
import visualisation


def getDataset(fichier):
    # data as numpy array
    data = np.genfromtxt(fichier, delimiter=',', skip_header=1)
    dimention = data.shape[1]
    return data, dimention


def saveResult(fichier, data, dimention):
    with open(fichier, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # header : x1, x2, x3, ... , centroid_x1, centroid_x2, centroid_x3, ... selon la dimention
        writer.writerow(["x" + str(i) for i in range(dimention)] + ["centroid_x" + str(i) for i in range(dimention)])
        for row in data:
            writer.writerow(row)

def kmeans(data, dimention, NB_CLUSTERS, MAX_ITER):
    #initialisation
    centroids = data[np.random.choice(data.shape[0], NB_CLUSTERS, replace=False), :]
    clusters = np.array([])
    iteration = 0
    convergence = False
    old_distance_moyenne = float('inf')
    distance_moyenne = float('inf')

    while not convergence and iteration < MAX_ITER:
        #creation des clusters
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # calcul des nouveaux centroids
        new_centroids = np.array([np.mean(data[clusters == i], axis=0) for i in range(NB_CLUSTERS)])

        # verification de la convergence
        distance_moyenne = np.mean(np.linalg.norm(centroids - new_centroids, axis=1))
        if distance_moyenne == old_distance_moyenne:
            convergence = True
        else:
            old_distance_moyenne = distance_moyenne
        distance_point_centroid = np.linalg.norm(data - centroids[clusters], axis=1)
        distance_moyenne_point_centroid = np.mean(distance_point_centroid)
        centroids = new_centroids
        iteration += 1

    # formattage pour la visualisation [[x1, y1, centroid_x, centroid_y], [x2, y2, centroid_x, centroid_y], ...]
    visualisation_data = np.concatenate((data, centroids[clusters]), axis=1)
    return visualisation_data, distance_moyenne_point_centroid

def runKmeans(data, dimention, NB_CLUSTERS, MAX_ITER, NB_RUNS):
    # run kmeans NB_run de fois et garde le meilleur resultat
    meilleur_resultat = None
    meilleure_distance_moyenne_point_centroids = float('inf')
    for i in range(NB_RUNS):
        visualisation_data, distance_moyenne_point_centroids = kmeans(data, dimention, NB_CLUSTERS, MAX_ITER)
        if distance_moyenne_point_centroids < meilleure_distance_moyenne_point_centroids:
            meilleure_distance_moyenne_point_centroids = distance_moyenne_point_centroids
            meilleur_resultat = visualisation_data
    print("moyenne de distance point_centroids " + str(meilleure_distance_moyenne_point_centroids))
    return meilleur_resultat

NB_CLUSTERS = 10
MAX_ITER = 100
NB_RUNS = 10

data, dimention = getDataset("./data/3d_data.csv")
out_data = runKmeans(data, dimention, NB_CLUSTERS, MAX_ITER, NB_RUNS)
if dimention == 2 or dimention == 3:
    visualisation.draw(out_data)
# save the result in a csv file
saveResult("./data/data_result.csv", out_data, dimention)