# # import the necessary packages
# from tensorflow.python.keras import layers, models
# from keras.layers.normalization import BatchNormalization
# # from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
# from keras import backend as K
#
#
# class MiniVGGNet:
#     @staticmethod
#     def build(width, height, depth, classes):
#         # initialize the model along with the input shape to be
#         # "channels last" and the channels dimension itself
#         model = models.Sequential()
#         inputShape = (height, width, depth)
#         chanDim = -1
#
#         # if we are using "channels first", update the input shape
#         # and channels dimension
#         if K.image_data_format() == "channels_first":
#             inputShape = (depth, height, width)
#             chanDim = 1
#
#         # first CONV => RELU => CONV => RELU => POOL layer set
#         model.add(layers.Conv2D(32, (3, 3), padding="same",
#                                 input_shape=inputShape))
#         model.add(layers.Activation("relu"))
#         model.add(BatchNormalization(axis=chanDim))
#         model.add(layers.Conv2D(32, (3, 3), padding="same"))
#         model.add(layers.Activation("relu"))
#         model.add(BatchNormalization(axis=chanDim))
#         model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#         model.add(layers.Dropout(0.25))
#
#         # second CONV => RELU => CONV => RELU => POOL layer set
#         model.add(layers.Conv2D(64, (3, 3), padding="same"))
#         model.add(layers.Activation("relu"))
#         model.add(BatchNormalization(axis=chanDim))
#         model.add(layers.Conv2D(64, (3, 3), padding="same"))
#         model.add(layers.Activation("relu"))
#         model.add(BatchNormalization(axis=chanDim))
#         model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#         model.add(layers.Dropout(0.25))
#
#         # first (and only) set of FC => RELU layers
#         model.add(layers.Flatten())
#         model.add(layers.Dense(512))
#         model.add(layers.Activation("relu"))
#         model.add(BatchNormalization())
#         model.add(layers.Dropout(0.5))
#
#         # softmax classifier
#         model.add(layers.Dense(classes))
#         model.add(layers.Activation("softmax"))
#
#         # return the constructed network architecture
#         return model

from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model