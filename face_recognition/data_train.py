import cv2
from sklearn import neighbors
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import numpy as np
import dlib
import pickle
from tqdm import tqdm

PREDICTOR_68_FACE = 'data_dlib/shape_predictor_68_face_landmarks.dat'

def train_face_recognition_model(data_dir='data', model_save_path='models/knn_model.clf', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat', n_neighbors=5):
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    # 初始化特征向量列表X和标签列表y
    X = []
    y = []

    # 人脸特征提取--ResNet
    face_rec_model = dlib.face_recognition_model_v1('data_dlib/dlib_face_recognition_resnet_model_v1.dat')
    # 人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 人脸对齐
    predictor = dlib.shape_predictor(predictor_path)

    for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)

        for person_name in os.listdir(identity_dir):
            person_dir = os.path.join(identity_dir, person_name)
            person_encodings = []

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                # img = cv2.imread(img_path)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"无法读取图像文件: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                # 遍历检测到的人脸
                for face in faces:
                    # 获取人脸关键点
                    shape = predictor(gray, face)
                    # 计算人脸特征向量--128维特征向量
                    encoding = np.array(face_rec_model.compute_face_descriptor(img, shape))
                    person_encodings.append(encoding)

                    # 数据增强：翻转图像并计算特征向量
                    flipped_face = cv2.flip(img, 1)
                    gray_flipped = cv2.cvtColor(flipped_face, cv2.COLOR_BGR2GRAY)
                    faces_flipped = detector(gray_flipped)

                    for face_flipped in faces_flipped:
                        shape_flipped = predictor(gray_flipped, face_flipped)
                        aligned_flipped_face = dlib.get_face_chip(flipped_face, shape_flipped, size=150)
                        flipped_encoding = np.array(face_rec_model.compute_face_descriptor(aligned_flipped_face))
                        person_encodings.append(flipped_encoding)
                    # shape = predictor(flipped_face, dlib.rectangle(0, 0, flipped_face.shape[1], flipped_face.shape[0]))
                    # aligned_flipped_face = dlib.get_face_chip(flipped_face, shape, size=224)
                    # flipped_encoding = np.array(face_rec_model.compute_face_descriptor(aligned_flipped_face))
                    # person_encodings.append(flipped_encoding)

                if person_encodings:
                    mean_encoding = np.mean(person_encodings, axis=0)
                    X.append(mean_encoding)
                    y.append(f"{identity}_{person_name}")

    if len(X) == 0:
        print("没有可用于训练的数据。请检查数据集。")
        return

    # KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    # 使用SVM分类器并进行数据标准化
    # clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', probability=True))
    # for i in tqdm(range(1), desc="Training"):
    #     clf.fit(X, y)
    #
    # with open(model_save_path, 'wb') as f:
    #     pickle.dump(clf, f)

# 训练模型
train_face_recognition_model(data_dir='data', model_save_path='models/knn_model.clf', predictor_path= PREDICTOR_68_FACE)
# train_face_recognition_model(data_dir='data', model_save_path='models/svm_model.clf', predictor_path= PREDICTOR_68_FACE)