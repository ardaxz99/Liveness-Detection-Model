# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50,VGG19,Xception,ResNet101,ResNet152,ResNet50V2
from tensorflow.keras.applications import ResNet152V2, InceptionResNetV2, ResNet101V2, DenseNet121, DenseNet169,DenseNet201
import tensorflow as tf

class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model


class Liveness_VGG16:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = VGG16(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_InceptionV3:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = InceptionV3(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet50:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape=input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_VGG19:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = VGG19(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_Xception:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = Xception(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet101:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet101(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet152:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet152(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet50V2:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet50V2(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet101V2:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet101V2(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_ResNet152V2:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet152V2(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model


class Liveness_InceptionResNetV2:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_DenseNet121:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = DenseNet121(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_DenseNet169:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = DenseNet169(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class Liveness_DenseNet201:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = DenseNet201(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model