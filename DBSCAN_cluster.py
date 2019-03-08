from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import cluster_helper as ch

COLORS = ['red', 'green', 'blue', 'cyan', 'pink', 'yellow', 'black', 'purple',
        'magenta', 'lime', 'chartreuse', 'olive', 'turquoise', 'salmon', 'teal',
        'lightblue', 'khaki', 'grey']

def dbscan_model(df):
    # Declaring Model
    dbscan = DBSCAN(eps=0.3, min_samples=5)

    # Fitting
    dbscan.fit(df)

    ch.save_classifier("spencer_dbscan", dbscan)

    # Transoring Using PCA
    pca = PCA(n_components=2).fit(df)
    pca_2d = pca.transform(df)
    c = [None] * len(COLORS)
    # Plot based on Class
    print("cluster")
    for i in range(0, pca_2d.shape[0]):
        print(dbscan.labels_[i])
        label = dbscan.labels_[i]
        if label < len(COLORS):
            c[label] = plt.scatter(pca_2d[i, 0], pca_2d[i, 1],
                    c=COLORS[label], marker='+')

    plt.legend(c, COLORS)
    plt.title('DBSCAN Clusters')
    return plt

if __name__ == '__main__':
    df = ch.featureCSV("user_playlist/Spencer.csv", ['name', 'album', 'id',
        'duration', 'key', 'mode', 'tempo', 'time_sig'])
    dbscan_plot = dbscan_model(df)
    ch.display_and_store(dbscan_plot,"dbscan_cluster.png")
