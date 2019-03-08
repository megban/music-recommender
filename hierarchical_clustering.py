# Importing Modules
from scipy.cluster.hierarchy import linkage, dendrogram, fclusterdata
import cluster_helper as ch
import csv


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
def h_clustering(varieties, samples, fname):
    if ch.classifier_exists(fname) == False:
        mergings = linkage(samples, method='complete')
        ch.save_classifier(fname, mergings)
        print("Successfully saved.")
    else:
        mergings = ch.load_classifier(fname)
        print("Successfully loaded.")


    """
    Plot a dendrogram using the dendrogram() function on mergings,
    specifying the keyword arguments labels=varieties, leaf_rotation=90,
    and leaf_font_size=6.
    """
    for i in range(0,len(mergings)):
        print(mergings[i])

    for i,v in zip(range(0,len(varieties)),varieties):
        print(str(i) + ". " + v)
    print(varieties[0])
    h_recommendations(mergings,varieties,"h_test.csv")
    print(fclusterdata(mergings, 5))
    return dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)

"""
model : mergings, h_cluster
"""
def h_recommendations(mergings , varieties, fname):
    with open(fname, 'w+') as csvfile:
        fieldnames = ['if you like:', 'you will like:']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m in mergings:
            if m[3]==2:
                writer.writerow({'if you like:': varieties[int(m[0])],
                                 'you will like:': varieties[int(m[1])]})

if __name__ == '__main__':
    varieties, samples = create_df("data/Playlist1.csv", ['id', 'album'], 'name')
    h_cluster = h_clustering(varieties, samples, "h_cluster_Playlist2")

    ch.display_and_store(h_cluster,"h_cluster.png")

