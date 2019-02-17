# Importing Modules
from scipy.cluster.hierarchy import linkage, dendrogram
import cluster_helper as ch

# import matplotlib.pyplot as plt
# import pandas as pd

def create_df(csv_path, to_drop, to_variety):
    # Reading the DataFrame
    df = ch.featureCSV(csv_path, to_drop)


    # Remove the name of songs from the DataFrame, save for later

    varieties = list(df.pop(to_variety))

    # Extract the measurements as a NumPy array
    samples = df.values
    return varieties, samples



"""
Perform hierarchical clustering on samples using the
linkage() function with the method='complete' keyword argument.
Assign the result to mergings.
"""
def h_clustering(varieties, samples):
    mergings = linkage(samples, method='complete')

    """
    Plot a dendrogram using the dendrogram() function on mergings,
    specifying the keyword arguments labels=varieties, leaf_rotation=90,
    and leaf_font_size=6.
    """
    return dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)


if __name__ == '__main__':
    varieties, samples = create_df("data/Playlist1.csv", ['id', 'album'], 'name')
    h_cluster = h_clustering(varieties, samples)
    ch.display_and_store(h_cluster,"h_cluster.png")

