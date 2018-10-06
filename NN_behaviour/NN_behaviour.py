import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import pandas as pd
from glob import glob
import re
import matplotlib.pyplot as plt

# Import libraries for keras
# # set the matplotlib backend so figures can be saved in the background
# import matplotlib
# matplotlib.use("Agg")
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
os.environ["KERAS_BACKEND"] = "theano"
import theano
import keras

# import the necessary packages
# from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
# from keras.optimizers import SGD
# from imutils import paths
# import argparse
import random
# import pickle
# import cv2
import os
import os.path

def check_file_path(file_path):
    if os.path.isfile(file_path):
        return file_path
    else:
        to_cluster_path = "/Dedicated/jmichaelson-" + file_path[1:]
        to_local_path = "/" + file_path[22:]

        if os.path.isfile(to_cluster_path):
            return to_cluster_path
        elif os.path.isfile(to_local_path):
            return to_local_path
        else:
            print("No valid file path")
            return "NOPE"
print("Load data...")
total_data_df_path = '/Dedicated/jmichaelson-wdata/mcrichter/HackUiowa2018/NN_behaviour/total_data_df_reduced_no_0_columns.csv'
total_data_df = pd.read_csv(check_file_path(total_data_df_path))

seed = 42
total_data_df_shuffled = shuffle(total_data_df, random_state=seed)


X = total_data_df_shuffled.drop(["intercept", "Score"], axis = 1)
y = total_data_df_shuffled[["Score"]]

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
y_scaled = min_max_scaler.fit_transform(y)

# Run the normalizer on the dataframe
y_normalized = pd.DataFrame(y_scaled, columns = ['Score_normalized'])

(trainX, testX, trainY, testY) = train_test_split(X, y_normalized, test_size=0.1, random_state=seed)


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))

print("Create model...")
model = Sequential()
model.add(Dense(10000, input_shape=(trainX.shape[1],), activation="sigmoid"))
model.add(Dense(5000, activation="sigmoid"))
model.add(Dense(500, activation="sigmoid"))
model.add(Dense(50, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))


# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 20
BATCH_SIZE = 30

# compile the model
print("Compile model...")
model.compile(loss='mean_squared_error', optimizer="SGD", metrics=["mae"])

print("Fit model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BATCH_SIZE)
print("Done")

########### PLOT ######################

# evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=BATCH_SIZE)
# print(classification_report(testY.argmax(axis=1),
# predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["mae"], label="train_mae")
plt.plot(N, H.history["val_mae"], label="val_mae")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.savefig(args["plot"])

########### TF STUFF

#
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))


# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config....)
#
# import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(
#   allow_soft_placement=True, log_device_placement=True))
