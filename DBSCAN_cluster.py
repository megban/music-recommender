from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import cluster_helper as ch

# Load Dataset


def dbscan_model(df):
    # Declaring Model
    dbscan = DBSCAN()

    # Fitting
    dbscan.fit(df)

    # Transoring Using PCA
    pca = PCA(n_components=2).fit(df)
    pca_2d = pca.transform(df)
    c1 = None
    c2 = None
    # Plot based on Class
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    if c1!=None and c2!=None:
        plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
        plt.title('DBSCAN finds 2 clusters and Noise')
    elif c1!=None:
        plt.legend([c1, c3], ['Cluster 1', 'Noise'])
        plt.title('DBSCAN finds 1 clusters and Noise')
    elif c2!=None:
        plt.legend([c2, c3], ['Cluster 1',  'Noise'])
        plt.title('DBSCAN finds 2 clusters and Noise')
    else:
        plt.title("Noise in cluster")
    return plt

if __name__ == '__main__':
    df = ch.featureCSV("data/unsupervised_data.csv", ['name', 'album', 'id','genre'])
    dbscan_plot = dbscan_model(df)
    ch.display_and_store(dbscan_plot,"dbscan_cluster.png")
