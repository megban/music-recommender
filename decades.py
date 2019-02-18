'''
Trains a model to classify seven decades of music playlists.
The decades we are training are 50s, 60s, 70s, 80s, 90s, 00s, 10s. (7 classes)
We've mapped class 0 to 2000, 1 to 2010, 2 to 1950, 3 to 1960, etc.

The accuracy is very poor -- the most I could get is 40% and without a little
coaxing it tends to perform on par with random (14%). This implies there is not
enough information in the data to distinguish between the classes.
'''
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tensorflow import keras
to_categorical = keras.utils.to_categorical
Sequential = keras.Sequential
Dense = keras.layers.Dense
Activation = keras.layers.Activation

FILENAME = 'data/Decades.csv'

data = pd.read_csv(FILENAME)
#data.drop(columns=['id', 'name', 'album'])
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns= ['id',
    'name', 'album', 'decade', 'duration', 'key', 'mode', 'tempo', 'time_sig']),
    data[['decade']], test_size = 0.1,
    random_state=0)

train_hot_labels, test_hot_labels = to_categorical(y_train, num_classes=7), to_categorical(y_test, num_classes=7)

model = Sequential([
    Dense(500, input_shape=(8,)), Activation('sigmoid'),
    Dense(300), Activation('sigmoid'),
    Dense(7), Activation('softmax'),
    ])

model.compile(optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

history = model.fit(X_train, train_hot_labels, validation_split=0.1,
        epochs=150, batch_size=32)

# Evaluate the model's performance
train_loss, train_acc = model.evaluate(X_train, train_hot_labels)
test_loss, test_acc = model.evaluate(X_test, test_hot_labels)

print('Training set accuracy:', train_acc)
print('Test set accuracy:', test_acc)

# The history of our accuracy during training.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
# Dimensions to slice the data for display
dims = ('danceability', 'energy', 'instrumentalness')
slices = []
for dim in dims:
    slices.append(train_data.get(dim))

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(slices[0], slices[1], slices[2], c=all_predictions)
plt.show()
"""
