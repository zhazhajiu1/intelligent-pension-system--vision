# # -*- coding: utf-8 -*-
# """
# 训练摔倒检测模型
# """
#
# # import the necessary packages
# from oldcare.datasets import SimpleDatasetLoader
# from oldcare.preprocessing import AspectAwarePreprocessor
# from oldcare.preprocessing import ImageToArrayPreprocessor
# from oldcare.conv import MiniVGGNet
# from oldcare.callbacks import TrainingMonitor
# from imutils import paths
# from sklearn.model_selection import train_test_split
# from keras._tf_keras.keras.optimizers import SGD
# # from keras.optimizers import SGD
# from sklearn.metrics import classification_report
# from keras._tf_keras.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder
# from keras_preprocessing.image import ImageDataGenerator
#
# # 全局变量
# dataset_path = 'dataset'
# output_model_path = 'models/fall_detection.hdf5'
# output_plot_path = 'plots/fall_detection.png'
#
# # 全局常量
# TARGET_WIDTH = 64
# TARGET_HEIGHT = 64
# BATCH_SIZE = 32
# EPOCHS = 10
# LR_INIT = 0.01
# DECAY = LR_INIT / EPOCHS
# MOMENTUM = 0.9
#
# # 加载图片
# aap = AspectAwarePreprocessor(TARGET_WIDTH, TARGET_HEIGHT)
# iap = ImageToArrayPreprocessor()
#
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(dataset_path))
#
# sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# (data, labels) = sdl.load(imagePaths, 500, False)
# data = data.astype("float") / 255.0
#
# # convert the labels from integers to vectors
# le = LabelEncoder().fit(labels)
# labels = to_categorical(le.transform(labels), 2)
#
# # partition the data into training and testing splits using 80% of
# # the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data,
#                                                   labels, test_size=0.20, stratify=labels, random_state=42)
#
# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                          height_shift_range=0.1, shear_range=0.2,
#                          zoom_range=0.2, horizontal_flip=True,
#                          fill_mode="nearest")
#
# # initialize the model
# print("[INFO] compiling model...")
# model = MiniVGGNet.build(width=TARGET_WIDTH, height=TARGET_HEIGHT,
#                          depth=3, classes=2)
# opt = SGD(lr=LR_INIT, decay=DECAY, momentum=MOMENTUM, nesterov=True)
# model.compile(loss="binary_cross entropy", optimizer=opt,
#               metrics=["accuracy"])
#
# # construct the set of callbacks
# callbacks = [TrainingMonitor(output_plot_path)]
#
# # train the network
# print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
#                         validation_data=(testX, testY),
#                         steps_per_epoch=len(trainX) // BATCH_SIZE,
#                         epochs=EPOCHS,
#                         callbacks=callbacks, verbose=1)
#
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=BATCH_SIZE)
# print(classification_report(testY.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=le.classes_))
#
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(output_model_path)
# -*- coding: utf-8 -*-
# """
# 训练摔倒检测模型
# """
#
# # import the necessary packages
# from keras._tf_keras.keras.applications import ResNet50
# from keras._tf_keras.keras.applications.resnet50 import preprocess_input
# from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
# from keras._tf_keras.keras.models import Model
# from oldcare.datasets import SimpleDatasetLoader
# from oldcare.preprocessing import AspectAwarePreprocessor
# from oldcare.preprocessing import ImageToArrayPreprocessor
# from oldcare.callbacks import TrainingMonitor
# from imutils import paths
# from sklearn.model_selection import train_test_split
# from keras._tf_keras.keras.optimizers import SGD
# from sklearn.metrics import classification_report
# from keras._tf_keras.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder
# from keras_preprocessing.image import ImageDataGenerator
#
# # 全局变量
# dataset_path = 'dataset'
# output_model_path = 'models/fall_detection.hdf5'
# output_plot_path = 'plots/fall_detection.png'
#
# # 全局常量
# TARGET_WIDTH = 64
# TARGET_HEIGHT = 64
# BATCH_SIZE = 32
# EPOCHS = 10
# LR_INIT = 0.01
# DECAY = LR_INIT / EPOCHS
# MOMENTUM = 0.9
#
# # 加载图片
# aap = AspectAwarePreprocessor(TARGET_WIDTH, TARGET_HEIGHT)
# iap = ImageToArrayPreprocessor()
#
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(dataset_path))
#
# sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# (data, labels) = sdl.load(imagePaths, 500, False)
# data = data.astype("float") / 255.0
#
# # convert the labels from integers to vectors
# le = LabelEncoder().fit(labels)
# labels = to_categorical(le.transform(labels), 2)
#
# # partition the data into training and testing splits using 80% of
# # the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data,
#                                                   labels, test_size=0.20, stratify=labels, random_state=42)
#
# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                          height_shift_range=0.1, shear_range=0.2,
#                          zoom_range=0.2, horizontal_flip=True,
#                          fill_mode="nearest")
#
# # initialize the model
# print("[INFO] compiling model...")
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(TARGET_WIDTH, TARGET_HEIGHT, 3))
#
# # Add your own classifier on top
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(2, activation='softmax')(x)
#
# model = Model(inputs=base_model.input, outputs=predictions)
#
# opt = SGD(lr=LR_INIT, decay=DECAY, momentum=MOMENTUM, nesterov=True)
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#
# # construct the set of callbacks
# callbacks = [TrainingMonitor(output_plot_path)]
#
# # train the network
# print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
#                         validation_data=(testX, testY),
#                         steps_per_epoch=len(trainX) // BATCH_SIZE,
#                         epochs=EPOCHS,
#                         callbacks=callbacks, verbose=1)
#
# # evaluate the network
# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=BATCH_SIZE)
# print(classification_report(testY.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=le.classes_))
#
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(output_model_path)
import cv2
'''
python trainingfalldetection.py
'''
# import the necessary packages
from keras._tf_keras.keras.applications import ResNet50
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from oldcare.datasets import SimpleDatasetLoader
from oldcare.preprocessing import AspectAwarePreprocessor
from oldcare.preprocessing import ImageToArrayPreprocessor
from oldcare.callbacks import TrainingMonitor
from imutils import paths
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras._tf_keras.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.image import ImageDataGenerator
import os
import glob
import numpy as np
# 全局变量
dataset_path = 'dataset'
output_model_path = 'models/fall_detection.hdf5'
output_plot_path = 'plots/fall_detection.png'


# 全局常量
TARGET_WIDTH = 64
TARGET_HEIGHT = 64
BATCH_SIZE = 32
EPOCHS = 10
LR_INIT = 0.01
DECAY = LR_INIT/EPOCHS
MOMENTUM = 0.9


# 加载图片
aap = AspectAwarePreprocessor(TARGET_WIDTH, TARGET_HEIGHT)
iap = ImageToArrayPreprocessor()

print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, 500, False)
data = data.astype("float") / 255.0

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# 全局变量
# dataset_path = 'dataset'
# output_model_path = 'models/fall_detection.hdf5'
# output_plot_path = 'plots/fall_detection.png'
#
# # 全局常量
# TARGET_WIDTH = 64
# TARGET_HEIGHT = 64
# BATCH_SIZE = 32
# EPOCHS = 10
# LR_INIT = 0.01
# DECAY = LR_INIT / EPOCHS
# MOMENTUM = 0.9
#
# # 获取图片路径
# fall_image_paths = glob.glob(os.path.join(dataset_path, 'fall', '*.png'))
# normal_image_paths = glob.glob(os.path.join(dataset_path, 'normal', '*.png'))
#
# # 合并两个列表
# imagePaths = fall_image_paths + normal_image_paths
#
# # 初始化预处理器
# aap = AspectAwarePreprocessor(TARGET_WIDTH, TARGET_HEIGHT)
# iap = ImageToArrayPreprocessor(dataFormat='channels_last')  # 或者 'channels_first'，根据你的需求
#
# print("[INFO] loading images...")
#
# sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# # print(imagePaths)
# # (data, labels) = sdl.load(imagePaths, 500, False)
# # data = data.astype("float") / 255.0
# # 加载图像并处理
# data = []
# labels = []
# for imagePath in imagePaths:
#     image = cv2.imread(imagePath)
#     if image is None:
#         print(f"[WARNING] Image {imagePath} could not be loaded, skipping...")
#         continue
#
#     image = aap.preprocess(image)
#     image = iap.preprocess(image)
#     data.append(image)
#
#     # 根据文件路径中的类别标签进行处理
#     label = 1 if 'fall' in imagePath else 0
#     labels.append(label)
#
# data = np.array(data)
# labels = np.array(labels)
#
# # 数据归一化
# data = data.astype("float") / 255.0
#
# # convert the labels from integers to vectors
# fall_labels = np.ones((len(fall_image_paths),), dtype="int")  # 摔倒类别为1
# normal_labels = np.zeros((len(normal_image_paths),), dtype="int")  # 日常活动类别为0
#
# labels = np.concatenate((fall_labels, normal_labels), axis=0)
# labels = to_categorical(labels, 2)  # 转换为独热编码

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Check the shapes of the split datasets
print(f"trainX shape: {trainX.shape}, trainY shape: {trainY.shape}")
print(f"testX shape: {testX.shape}, testY shape: {testY.shape}")

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(TARGET_WIDTH, TARGET_HEIGHT, 3))

# Add your own classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

opt = SGD(learning_rate=LR_INIT, decay=DECAY, momentum=MOMENTUM, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["acc"])#TODO:acc/accuracy

# construct the set of callbacks
callbacks = [TrainingMonitor(output_plot_path)]

# train the network
print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=callbacks, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=['normal', 'fall']))

# save the model to disk
print("[INFO] serializing network...")
model.save(output_model_path)

