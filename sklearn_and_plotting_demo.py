# Tutorial from
# https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03
# Hello
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading dataset
iris_df = datasets.load_iris()

# Available methods on dataset
print(dir(iris_df))

# Features
print(iris_df.feature_names)

# Targets
print(iris_df.target)

# Target Names
print(iris_df.target_names)
label = {0: 'red', 1: 'blue', 2: 'green'}

# Dataset Slicing
x_axis = iris_df.data[:, 1]
y_axis = iris_df.data[:, 2]
z_axis = iris_df.data[:, 3]

fig = plt.figure(1)
ax = fig.add_subplot(211, projection='3d')

ax.scatter(x_axis, y_axis, z_axis, c=iris_df.target)

model = KMeans(n_clusters=3)
model.fit(iris_df.data)

# Predicitng a single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# Prediction on the entire data
all_predictions = model.predict(iris_df.data)

# Printing Predictions
print(predicted_label)
print(all_predictions)

ax = fig.add_subplot(212, projection='3d')
ax.scatter(x_axis, y_axis, z_axis, c=all_predictions)
plt.show()
