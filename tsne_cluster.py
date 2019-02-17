# Importing Modules

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cluster_helper as ch
# Loading dataset



# Defining Model
def tsne_model(df):
    model = TSNE(learning_rate=100)

    # Fitting Model
    transformed = model.fit_transform(df)

    # Plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    return plt.scatter(x_axis, y_axis)


if __name__ == '__main__':
    df = ch.featureCSV("data/Playlist1.csv", ['name','album','id'])
    scatter_plot = tsne_model(df)
    ch.display_and_store(scatter_plot, "tsne_cluster")