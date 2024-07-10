import os
from sklearn import neighbors
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

from tqdm import tqdm
from algorithm.face_recognition.data_preparation import preprocess_data


# KNN
def train_knn_model(data_dir='data', model_save_path='models/knn_model.clf', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat', n_neighbors=5):
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    print("knn数据预处理...")
    X, y = preprocess_data(data_dir, predictor_path)

    if len(X) == 0:
        print("没有可用于训练的数据。请检查数据集。")
        return

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    print("训练KNN model...")
    for i in tqdm(range(1), desc="Training"):
        knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

# SVM
def train_svm_model(data_dir='data', model_save_path='models/svm_model.clf', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat'):
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    print("SVM数据预处理...")
    X, y = preprocess_data(data_dir, predictor_path)

    if len(X) == 0:
        print("没有可用于训练的数据。请检查数据集。")
        return

    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', probability=True))
    print("训练SVM model...")
    for i in tqdm(range(1), desc="Training"):
        clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)


# 训练 KNN 模型
train_knn_model(data_dir='data', model_save_path='models/knn_model.clf', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat')

# 训练 SVM 模型
train_svm_model(data_dir='data', model_save_path='models/svm_model.clf', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat')

