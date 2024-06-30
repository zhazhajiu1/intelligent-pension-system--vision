import os
import cv2
import dlib
import numpy as np
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# 全局路径变量
DATA_DIR = 'data'
DATA_TEST_DIR = 'data_test'
MODEL_DIR = 'models'
PREDICTOR_PATH = 'data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_REC_MODEL_PATH = 'data_dlib/dlib_face_recognition_resnet_model_v1.dat'
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.clf')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.clf')

# 全部预测
def evaluate_model(model_path, data_dir, predictor_path):
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 初始化 Dlib 的人脸检测器和特征预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1('data_dlib/dlib_face_recognition_resnet_model_v1.dat')

    y_true = []
    y_pred = []

    for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)

        for person_name in os.listdir(identity_dir):
            person_dir = os.path.join(identity_dir, person_name)

            i = 0
            j = 0

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"无法读取图像文件: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    shape = predictor(gray, face)
                    aligned_face = dlib.get_face_chip(img, shape, size=150)
                    encoding = np.array(face_rec_model.compute_face_descriptor(aligned_face))

                    probabilities = model.predict_proba([encoding])
                    max_prob = np.max(probabilities)
                    if max_prob >= 0.5:
                        label = model.predict([encoding])[0]
                        # print(f'识别出: {identity}_{person_name}{i}')
                        print(f'{identity}_{person_name}{i} 的预测标签: {label}')
                        i = i+1
                    else:
                        label = "Unknown"
                        print(f'未能识别出!!! Unknown{j}')
                        j = j+1

                    y_true.append(f"{identity}_{person_name}")
                    y_pred.append(label)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model accuracy: {accuracy}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return y_true, y_pred

# 预测指定
def predict_images_in_folder(model_path, folder_path, predictor_path, face_rec_model_path, threshold=0.5):
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 初始化 Dlib 的人脸检测器和特征预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    y_true = []
    y_pred = []

    if not os.path.isdir(folder_path):
        print(f"{folder_path} 不是一个有效的目录")
        return

    person_name = os.path.basename(folder_path)

    i = 0
    j = 0

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if not os.path.isfile(img_path):
            continue  # 跳过非文件项

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取图像文件: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            aligned_face = dlib.get_face_chip(img, shape, size=150)
            encoding = np.array(face_rec_model.compute_face_descriptor(aligned_face))

            # 检查并调整 n_neighbors
            if hasattr(model, 'n_neighbors'):
                if model.n_neighbors > model.n_samples_fit_:
                    model.n_neighbors = model.n_samples_fit_

            probabilities = model.predict_proba([encoding])
            max_prob = np.max(probabilities)
            if max_prob >= threshold:
                label = model.predict([encoding])[0]
                print(f'{person_name}{i} 的预测标签: {label}')
                i += 1
            else:
                label = "Unknown"
                print(f'未能识别出!!! Unknown{j}')
                j += 1

            y_true.append(person_name)
            y_pred.append(label)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Model accuracy: {accuracy}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return y_true, y_pred

# 绘制ROC曲线和AUC值
def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# 获取预测概率
def get_probabilities(model_path, data_dir, predictor_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1('data_dlib/dlib_face_recognition_resnet_model_v1.dat')

    y_true = []
    y_pred_proba = []

    for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)

        for person_name in os.listdir(identity_dir):
            person_dir = os.path.join(identity_dir, person_name)

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"无法读取图像文件: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    shape = predictor(gray, face)
                    aligned_face = dlib.get_face_chip(img, shape, size=150)
                    encoding = np.array(face_rec_model.compute_face_descriptor(aligned_face))

                    probabilities = model.predict_proba([encoding])
                    y_true.append(f"{identity}_{person_name}")
                    y_pred_proba.append(np.max(probabilities))

    return y_true, y_pred_proba

# PREDICT_DIR = 'data_test/elderly/lsy'

# 评估KNN模型
print("评估KNN model:")
# 全部
y_true_knn, y_pred_knn = evaluate_model('models/knn_model.clf', DATA_TEST_DIR, 'data_dlib/shape_predictor_68_face_landmarks.dat')
# 指定
# predict_images_in_folder(KNN_MODEL_PATH, PREDICT_DIR, PREDICTOR_PATH, FACE_REC_MODEL_PATH)

# 评估SVM模型
print("评估SVM model:")
# 全部
y_true_svm, y_pred_svm = evaluate_model('models/svm_model.clf', DATA_TEST_DIR, 'data_dlib/shape_predictor_68_face_landmarks.dat')
# 指定
# predict_images_in_folder(SVM_MODEL_PATH, PREDICT_DIR, PREDICTOR_PATH, FACE_REC_MODEL_PATH)

# 获取KNN和SVM的预测概率
# y_true_knn, y_pred_proba_knn = get_probabilities('models/knn_model.clf', DATA_TEST_DIR, 'data_dlib/shape_predictor_68_face_landmarks.dat')
# y_true_svm, y_pred_proba_svm = get_probabilities('models/svm_model.clf', DATA_TEST_DIR, 'data_dlib/shape_predictor_68_face_landmarks.dat')
# 绘制ROC曲线
# plot_roc_curve(y_true_knn, y_pred_proba_knn, "KNN ROC Curve")
# plot_roc_curve(y_true_svm, y_pred_proba_svm, "SVM ROC Curve")

