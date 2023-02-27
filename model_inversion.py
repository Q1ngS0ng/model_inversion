import tensorflow as tf
import pickle
import numpy as np
from easydict import EasyDict
import yaml
import os
import matplotlib.pyplot as plt

def load_config(config_name, config_path="./config"):
    # Read config.yaml file
    with open(config_path) as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG[config_name])
    return CFG

CFG = load_config("CFG")

X = pickle.load(open(CFG.X_path, "rb"))
y = pickle.load(open(CFG.y_path,"rb"))
XT = pickle.load(open(CFG.XT_path, "rb"))
yt = pickle.load(open(CFG.yt_path,"rb"))
X = X/255.0
XT = XT/255.0
IMG_X_SIZE = CFG.IMG_X_SIZE
IMG_Y_SIZE = CFG.IMG_Y_SIZE
names = os.listdir(CFG.TRAINDIR)

black_image_tensor = tf.convert_to_tensor(np.zeros((1, IMG_X_SIZE, IMG_Y_SIZE, 1)))

def show_img(im):  # defines a function to output the image
    plt.imshow(im,
               cmap='gray')  # take an array of 0-255 brightness values for each pixel in the image, and create a viewable object
    plt.show()  # display the image in the output

def show_img_label(im, label):  # defines a function to print a label, then output an image
    print(label)
    show_img(im)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def sarahsinversion(model, img, learning_rate, label, best_loss, best_img, counter):
    print(img[0].shape)
    show_img_label(tf.reshape(img[0], (IMG_Y_SIZE, IMG_X_SIZE)), "Starting Image:")
    with tf.GradientTape() as tape:
        tape.watch(img)
        prediction = model(img, training=False)  # run img through the model
        loss = loss_object(label, prediction)  # calculate the loss of img
    gradient = tape.gradient(loss, img)  # calculate the gradient with respect two each pixel in img
    print("Current Loss:", loss.numpy())

    img = tf.clip_by_value(img - learning_rate * gradient, 0, 255)
    show_img_label(tf.reshape(img[0], (IMG_Y_SIZE, IMG_X_SIZE)), "Updated Image:")

    img = np.array([np.clip(x + np.random.normal(2, 2), 0, 255) for x in img.numpy()])
    show_img_label(tf.reshape(img[0], (IMG_Y_SIZE, IMG_X_SIZE)), "Noise:")
    img = tf.convert_to_tensor(img)
    return img
import keras
model = keras.models.load_model("./models/lenet.h5")


for name_index in range(len(names)):
    if names[name_index] == "s2":
        print("Decoding: ", names[name_index])
        show_img_label(tf.reshape(XT[name_index], (IMG_Y_SIZE, IMG_X_SIZE)), "Goal Image:")
        best_img = black_image_tensor
        best_loss = float('inf')
        for i in range(10):
            best_img = sarahsinversion(model, best_img, 0.1, name_index, best_img, best_loss, i)
        break