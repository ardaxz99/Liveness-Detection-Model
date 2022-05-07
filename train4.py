# USAGE
# python train.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet, Liveness_VGG16, Liveness_InceptionV3, Liveness_ResNet50,Liveness_VGG19,Liveness_Xception,Liveness_ResNet101,Liveness_ResNet152,Liveness_ResNet50V2
from pyimagesearch.livenessnet import Liveness_ResNet152V2, Liveness_InceptionResNetV2, Liveness_ResNet101V2, Liveness_DenseNet121, Liveness_DenseNet169,Liveness_DenseNet201
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import time
from sklearn.utils import class_weight
from keras.models import load_model


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 100
EPOCHS = 50
img_height=224
img_width=224

start = time.time()

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model = Liveness_VGG16.build(width=img_width, height=img_height, depth=3,
	classes=2)
"""
model = load_model("liveness.model")
"""
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


valid_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(
    directory="C:\\Users\\ARDA  BARIŞ\\Desktop\\facial recog\\liveness-detection-opencv\\liveness-detection-opencv\\output\\train",
    target_size=(img_height, img_width),
    batch_size=BS,
    seed=42,
	shuffle=True,
    class_mode='categorical')
 # set as training data

validation_generator = valid_datagen.flow_from_directory(
    directory="C:\\Users\\ARDA  BARIŞ\\Desktop\\facial recog\\liveness-detection-opencv\\liveness-detection-opencv\\output\\val", # same directory as training data
    target_size=(img_height, img_width),
    batch_size=BS,
	shuffle=False,
    class_mode='categorical')
# set as validation data

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes),
                train_generator.classes)
class_weights = {i : class_weights[i] for i in range(2)}

#Training
H=model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BS,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // BS,
	max_queue_size=100,
	workers = 8 ,# (set a proper value > 1)
    class_weight=class_weights,
    epochs = EPOCHS)


# evaluate the network
print("[INFO] evaluating network...")
predictions=model.predict(validation_generator,steps=np.ceil(validation_generator.samples // BS)+1)
print(classification_report(validation_generator.classes,predictions.argmax(axis=1),digits=4,target_names=["fake","real"]))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

end = time.time()
print(end - start)

