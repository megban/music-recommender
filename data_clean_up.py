from os.path import isfile, join
from os import listdir
import pandas as pd
onlyfiles = [f for f in listdir("data/") if isfile(join("data/", f))]
frames = []
for f in onlyfiles:
    frame = pd.read_csv("data/"+f)
    frame['genre'] = f.split(".")[0]
    frames.append(frame)
final_frame = pd.concat(frames)
final_frame.drop_duplicates(subset=['id'], keep=False)
final_frame.to_csv("unsupervised_data.csv")
