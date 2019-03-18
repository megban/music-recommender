import os
import zipfile
import shutil
import cluster_helper as ch
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '/run/media/spencer/Elements/Documents/CMPS144/img2048nolog'
genres = ['acoustic', 'classical', 'disco', 'electronic', 'hip-hop', 'indie', 'jazz', 'pop', 'r&b']

#fnames = {g: list(map(lambda x: os.path.join(base_dir,x), os.listdir(os.path.join(base_dir, g))[:200])) for g in genres}

#print(fnames['acoustic'][:10])
#exit(0)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras import layers
from tensorflow.keras import Model

def generators():
    # Data generation

    # All images will be rescaled by 1./255
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.33)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = datagen.flow_from_directory(
            base_dir,  # This is the source directory for training images
            target_size = (1292//4, 1024//4),  # All images will be resized to 150x150
            batch_size = 20,
            class_mode = 'categorical',
            subset = 'training')

    validation_generator = datagen.flow_from_directory(
            base_dir,  # This is the source directory for training images
            target_size = (1292//4, 1024//4),  # All images will be resized to 150x150
            batch_size = 20,
            class_mode = 'categorical',
            subset = 'validation')

    return train_generator, validation_generator


def model_and_history(train_generator, validation_generator):

    # Input is 1292x1024x3
    img_input = layers.Input(shape=(1292//4, 1024//4, 3))

    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor so we can add fully connected layers
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)

    x = layers.Dropout(0.75)(x)

    # Create output layer with 9 softmax nodes, one for each category.
    output = layers.Dense(9, activation='softmax')(x)

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer
    model = Model(img_input, output)

    """Let's summarize the model architecture:"""

    model.summary()

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = RMSprop(lr = 0.0001),
                  metrics = ['acc'])

    history = model.fit_generator(
            train_generator,
            steps_per_epoch = 100,  # 2000 images = batch_size * steps
            epochs = 10,
            validation_data = validation_generator,
            validation_steps = 50,  # 1000 images = batch_size * steps
            verbose = 2,
            use_multiprocessing=True)

    return model, history

if __name__ == '__main__':
    train_set, val_set = generators()
    model, history = model_and_history(train_set, val_set)
    # Evaluation

    # Retrieve a list of accuracy results on training and test data
    # sets for each training epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    print(acc)
    print(val_acc)

    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    # Eval labels
    #pred_label = model.predict_generator(validation_generator,
    #        workers=8, use_multiprocessing=True)

    #print('######### Labels #########')
    #print(pred_label)
