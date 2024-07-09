import atexit
import signal
import sys
import cv2
import dlib
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

PREDICTOR_PATH = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'
SKIP_FRAMES = 1

class EmotionRecognition:
    def __init__(self, model_path, predictor_path, face_rec_model_path):
        # 加载表情识别模型
        self.model = load_model(model_path)
        # 初始化 Dlib 的人脸检测器和特征预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

        self.emotions = ['happy', 'normal', 'surprise']

    def recognize_emotion(self, frame, face):
        shape = self.predictor(frame, face)
        aligned_face = dlib.get_face_chip(frame, shape, size=48)
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        aligned_face = aligned_face.reshape(1, 48, 48, 1).astype('float32') / 255.0

        emotion_prediction = self.model.predict(aligned_face)
        max_index = np.argmax(emotion_prediction)
        emotion = self.emotions[max_index]

        return emotion

def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()
    print("释放摄像机，关闭所有窗口.")

def signal_handler(sig, frame):
    print("收到信号，正常退出...")
    sys.exit(0)

def main():
    model_path = 'emotion_recognition_model.h5'
    predictor_path = PREDICTOR_PATH
    face_rec_model_path = '../face_recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
    face_recognition_model_path = '../face_recognition/models/knn_model.clf'

    emotion_recognition = EmotionRecognition(model_path, predictor_path, face_rec_model_path)

    # 加载训练好的 KNN 模型
    with open(face_recognition_model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # 初始化 Dlib 的人脸检测器和特征预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: 无法打开视频设备.")
        return

    # 寄存器清理
    atexit.register(release_resources, cap)
    signal.signal(signal.SIGINT, signal_handler)  # Catch Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Catch termination signal

    frame_count = 0  # 帧计数器

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: 捕获图像失败.")
                break

            # 每隔 skip_frames 帧处理一次
            if frame_count % SKIP_FRAMES == 0:
                # 将帧转换为灰度图以进行人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    shape = predictor(gray, face)
                    encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

                    # KNN
                    closest_distances = knn_clf.kneighbors([encoding], n_neighbors=1)
                    is_recognized = closest_distances[0][0][0] <= 0.4

                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    if is_recognized:
                        label = knn_clf.predict([encoding])[0]
                        identity, person_name = label.split('_')
                        color = (0, 255, 0)  # 绿色框表示识别成功
                        text = f"{identity}: {person_name}"
                    else:
                        color = (0, 0, 255)  # 红色框表示识别失败
                        text = "Unknown"

                    # 表情识别
                    emotion = emotion_recognition.recognize_emotion(frame, face)

                    # 在人脸周围绘制矩形框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # 在人脸上方显示表情标签
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # 在人脸下方显示身份标签
                    cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 显示结果帧
            cv2.imshow('Face and Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1  # 增加帧计数器
    except KeyboardInterrupt:
        print("进程被用户中断.")
    finally:
        release_resources(cap)

if __name__ == '__main__':
    main()
