# Impl. clustering

import csv
import sys
from typing import List

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
from joblib import dump, load

# Loads a CSV file for data use, stripping labels for songs
def featureCSV(filename: str):
    data = pd.read_csv(filename)
    idFeatures = data[['id','name']].copy()
    dataFeatures = data.drop(columns=['id', 'name', 'album'])
    return idFeatures, dataFeatures



# Trains a k-cluster model and displays the results
def cluster_and_show(songID, train_data, clusters: int):
    model = KMeans(n_clusters=clusters)
    #model = MiniBatchKMeans(n_clusters=clusters, random_state=0, batch_size=6)
    model.fit(train_data)

    dump(model, 'filename.kcluster')


    all_predictions = model.predict(train_data)
    songID['clusters'] = all_predictions

    # Dimensions to slice the data for display
    dims = ('danceability', 'energy', 'instrumentalness')
    slices = []
    for dim in dims:
        slices.append(train_data.get(dim))

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slices[0], slices[1], slices[2], c=all_predictions)
    ax.set_xlabel('danceability')
    ax.set_ylabel('energy')
    ax.set_zlabel('instrumentalness')
    plt.show()

    return all_predictions


def predictSong(songID):
    clusterStor = []
    for i in range(0,6):
        clusterStor.append(songID.loc[songID['clusters'] == i])

    sampleSongs = []
    for i in range(0,6):
        cluster = clusterStor[i]
        #sampleSongs.append(cluster.sample())
        print(cluster.sample())
    print("\nChoose a song cluster you like (0-6):")

    clusterAns = int(input())
    print(clusterStor[clusterAns])
    



if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Missing argument: CSV file")
        exit(1)

    # Load the dataset
    print("Loading {}".format(sys.argv[1]))
    songID, data = featureCSV(sys.argv[1])
    print("Done loading")

    # Display Predictions
    print("Training model...")
    cluster_and_show(songID,data, 7)

    predictSong(songID)

