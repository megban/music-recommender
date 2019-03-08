import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path

def display_and_store(plot, image_fname):
    plt.savefig(image_fname)
    plt.show()

def featureCSV(filename: str, to_drop):
    data = pd.read_csv(filename)
    return data.drop(columns=to_drop)


# Save to file in the current working directory
def save_classifier(fname, model):
    with open(fname + ".pkl", 'wb') as file:
        pickle.dump(model, file)

# Load from file


def classifier_exists(name):
    fname = Path(name + ".pkl")
    return fname.is_file()


def load_classifier(name):
    fname = Path(name+".pkl")

    with open(fname, 'rb') as file:
        pickle_model = pickle.load(file)
        return pickle_model

