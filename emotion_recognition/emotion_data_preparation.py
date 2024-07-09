import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class CKPlusDataset:
    def __init__(self, data_dir, image_size=(48, 48)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.emotions = {
            'happy': 0,
            'normal': 1,
            'surprise': 2,
            # 'anger': 0,
            # 'contempt': 1,
            # 'disgust': 2,
            # 'fear': 3,
            # 'happy': 4,
            # 'neutral': 5,
            # 'sadness': 6,
            # 'surprise': 7,
        }
        # 存储加载的数据和标签
        self.X, self.y = self.load_data()

    def load_data(self):
        X = []
        y = []
        for emotion, label in self.emotions.items():
            emotion_dir = os.path.join(self.data_dir, emotion)
            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # 灰度处理
                img = cv2.resize(img, self.image_size) #调整大小
                X.append(img)
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], self.image_size[0], self.image_size[1], 1)
        # 将图像像素值归一化到 [0, 1] 范围
        X = X.astype('float32') / 255.0
        y = to_categorical(y, num_classes=len(self.emotions))
        return X, y

    def get_train_test_split(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
