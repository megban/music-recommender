import matplotlib.pyplot as plt
import pandas as pd


def display_and_store(plot, image_fname):
    plt.savefig(image_fname)
    plt.show()

def featureCSV(filename: str, to_drop):
    data = pd.read_csv(filename)
    return data.drop(columns=to_drop)
