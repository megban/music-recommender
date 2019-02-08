# Impl. clustering

import csv
import sys
from typing import List

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Loads a CSV file for data use, stripping labels for songs
def featureCSV(filename: str):
    data = pd.read_csv(filename)
    return data.drop(columns=['id', 'name', 'album'])

# Trains a k-cluster model and displays the results
def cluster_and_show(train_data, clusters: int):
    model = KMeans(n_clusters=clusters)
    model.fit(train_data)

    all_predictions = model.predict(train_data)

    # Dimensions to slice the data for display
    dims = ('danceability', 'energy', 'instrumentalness')
    slices = []
    for dim in dims:
        slices.append(train_data.get(dim))

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slices[0], slices[1], slices[2], c=all_predictions)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("Missing argument: CSV file")
        exit(1)

    # Load the dataset
    print("Loading {}".format(sys.argv[1]))
    data = featureCSV(sys.argv[1])
    print("Done loading")

    # Display Predictions
    print("Training model...")
    cluster_and_show(data, 3)
