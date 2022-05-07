from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
import tensorflow as tf

class LivenessNet_VGG16:
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

class LivenessNet_InceptionV3:
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

class LivenessNet_ResNet50:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height,width,depth)
        model = ResNet50(weights="imagenet", include_top=False, input_shape= input_shape)
        model.trainable = False
        inputs = tf.keras.Input(shape= input_shape)
        x = model(inputs, training=False)
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        outputs = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model

class LivenessNet_VGG19:
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

class LivenessNet_Xception:
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

class LivenessNet_ResNet101:
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

class LivenessNet_ResNet152:
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

class LivenessNet_ResNet50V2:
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

class LivenessNet_ResNet101V2:
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

class LivenessNet_ResNet152V2:
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

class LivenessNet_InceptionV3:
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

class LivenessNet_InceptionV3:
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

class LivenessNet_InceptionResNetV2:
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

class LivenessNet_DenseNet121:
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

class LivenessNet_DenseNet169:
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

class LivenessNet_DenseNet201:
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