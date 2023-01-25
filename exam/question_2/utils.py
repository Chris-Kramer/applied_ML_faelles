
###########
# Imports #
###########
import tensorflow as tf

# data
import pandas as pd
import numpy as np
import tensorflow_datasets  as tfds

# Plotting
import matplotlib.pyplot as plt

########
# Data #
########
# ----- Helper functions -----
def _convert_sample(sample: pd.DataFrame,
                    gray_scale: bool = False,
                    size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    Desc
    -----
    Convert the pandas DataFrame into a list of images and a list of one hot encoding labels
    The Images can also be converted to gray_scale and resized if necessary
    
    Return
    -------
    pandas (images)
    list of numpy arrays (labels)
    '''
    # Get data
    image, label = sample['image'], sample['label']  
    
    # Convert image to tensor
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Convert to grayscale if necessary
    if gray_scale:
        image = tf.image.rgb_to_grayscale(image)
    # Resize image
    if size is not None:
        image = tf.image.resize(image, size)
    # Transform to one-hot-encodign labels
    label = tf.one_hot(label, 2, dtype=tf.float32)
    return image, label

def load_data(data_dir: str,
              perc: int = 5,
              size: tuple[int, int] | None = None,
              gray_scale: bool = False,
              batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Desc
    -----
    Returns a tuple of the train, test and validation data in a 80% 10% 10%
    
    Params
    -------
    data_dir: The path to the data directory
    perc: the percentage of the full data
    
    Return
    -------
    Tuple of numpay
    '''
    # Load data set
    train_df, val_df, test_df = tfds.load('patch_camelyon',split=[f'train[:{perc}%]', f'test[:{perc}%]', f'validation[:{perc}%]'],
                                          data_dir = data_dir,
                                          download=False,
                                          shuffle_files=True)
    
    # Convert _to numpy
    train_df = train_df.map(_convert_sample).batch(batch_size)
    val_df = val_df.map(_convert_sample).batch(batch_size)
    test_df = test_df.map(_convert_sample).batch(batch_size)        

    
    return (train_df, val_df, test_df)

############
# Plotting #
############
def plot_hist(history):
    '''
    Desc
    -----
    Plots the accuracy and loss for training and validation data
    '''
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    figure, axis = plt.subplots(2, 1) # display two plots in one graph

    axis[0].plot(epochs, acc)
    axis[0].plot(epochs, val_acc)
    axis[0].set_title('Training and validation accuracy')
    plt.legend()

    axis[1].plot(epochs, loss)
    axis[1].plot(epochs, val_loss)
    axis[1].set_title('Training and validation loss')
    plt.legend()

    plt.show()
    
