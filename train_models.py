import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

TRAINDIR = 'faces/train'  # directory where we store our data
IMG_Y_SIZE = 112
IMG_X_SIZE = 92
NUM_NAMES = 40
names = os.listdir(TRAINDIR)  # list of classes i.e. names


def show_img(im):  # defines a function to output the image
    plt.imshow(im,
               cmap='gray')  # take an array of 0-255 brightness values for each pixel in the image, and create a viewable object
    plt.show()  # display the image in the output


def show_img_label(im, label):  # defines a function to print a label, then output an image
    print(label)
    show_img(im)

training_data = []

def create_training_data():
    for name in names:
        name_num = names.index(name)
        for img in os.listdir(f"{TRAINDIR}/{name}"):
            img_array = cv2.imread(f"{TRAINDIR}/{name}/{img}",
                                   cv2.IMREAD_GRAYSCALE)  # convert the image to its pixel (brightness value) data
            new_array = cv2.resize(img_array, (IMG_X_SIZE, IMG_Y_SIZE))  # resize if need be
            training_data.append([new_array, name_num])  # add it to training_data along with its name

create_training_data()

random.shuffle(training_data)  # randomize the training data (make learning more effective)

# prepare the training data for ML i.e. convert to numpy

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_X_SIZE,IMG_Y_SIZE,1)
y = np.array(y)

print(X.shape, y.shape)

import pickle

# save your data (so you don't have to load it every time)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

TESTDIR = 'faces/test' # directory where we store our testing data

# load the testing data


testing_data = []

def create_testing_data():
    for name in names:
        name_num = names.index(name)
        for img in os.listdir(f"{TESTDIR}/{name}"):
            img_array = cv2.imread(f"{TESTDIR}/{name}/{img}",cv2.IMREAD_GRAYSCALE) # convert the image to its pixel (brightness value) data
            new_array = cv2.resize(img_array,(IMG_X_SIZE,IMG_Y_SIZE)) # resize if need be
            testing_data.append([new_array, name_num]) # add it to testing_data along with its name

create_testing_data()

XT = []
yt = []

for features, label in testing_data:
    XT.append(features)
    yt.append(label)

XT = np.array(XT).reshape(-1, IMG_X_SIZE, IMG_Y_SIZE, 1)
yt = np.array(yt)
print(XT.shape, yt.shape)

# save your data (so you don't have to load it every time)

pickle_out = open("XT.pickle","wb")
pickle.dump(XT, pickle_out)
pickle_out.close()


pickle_out = open("yt.pickle","wb")
pickle.dump(yt, pickle_out)
pickle_out.close()

import tensorflow as tf
#GPU test and settings
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)
#gpu_optoins = tf.GPUOptions(per_process_gpu_memory_fraction=0.3333)
#sess=tf.Session(config=tf.ConfigProto(gpu_options=gpuoptions))

import pickle
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle","rb"))
XT = pickle.load(open("XT.pickle", "rb"))
yt = pickle.load(open("yt.pickle","rb"))
X = X/255.0
XT = XT/255.0

train_ds = tf.data.Dataset.from_tensor_slices(
    (X, y)).shuffle(10000).batch(10)

test_ds = tf.data.Dataset.from_tensor_slices((XT, yt)).batch(10)

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,  Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

import time

# tensorflow is the most popular supported framework by all MLaaS service providers

NAME = "Face-Recognition-CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(filters=36, kernel_size=7, activation='relu', input_shape= (IMG_X_SIZE,IMG_Y_SIZE, 1))
    self.flatten = Flatten()
    self.d1 = Dense(NUM_NAMES)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return x

class Lenet(Model):
  def __init__(self):
    super(Lenet, self).__init__()
    self.conv1 = Conv2D(filters=64, kernel_size=7, activation='relu', input_shape= (IMG_X_SIZE,IMG_Y_SIZE, 1))
    self.flatten = Flatten()
    self.d1 = Dense(100)
    self.d2 = Dense(NUM_NAMES)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return x

def Keras_lenet():
    input = Input(shape=(IMG_X_SIZE,IMG_Y_SIZE, 1), name='input')
    x = Conv2D(64, 7, activation="relu")(input)
    x = Flatten()(x)
    x = Dense(100)(x)
    output = Dense(NUM_NAMES)(x)

    model = Model(inputs=[input], outputs=[output], name='lenet')
    return model

# Create an instance of the model
model = Lenet()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
#%%
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
#%%


EPOCHS = 10
tf.config.run_functions_eagerly(True)

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

import tensorflow as tf

# config = model.get_config()
weight = model.get_weights()


model = Keras_lenet()
model.set_weights(weight)
model.save("./models/{name}.h5".format(name = model.name))
