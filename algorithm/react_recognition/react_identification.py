import cv2
import dlib
import numpy as np
import pickle
from sklearn import neighbors

PREDICTOR_PATH = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = '../face_recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
KNN_MODEL_PATH = '../face_recognition/models/knn_model.clf'
SKIP_FRAMES = 1
DISTANCE_THRESHOLD = 1.0  # 1米内,判断为交互

class InteractionDetection:
    def __init__(self, knn_model_path, predictor_path, face_rec_model_path, distance_threshold):
        self.knn_clf = self.load_model(knn_model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        self.distance_threshold = distance_threshold
        self.volunteer_label = "volunteer"
        self.elderly_label = "elderly"

    # 加载训练好的模型(KNN)
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # 计算距离
    def calculate_distance(self, shape1, shape2):
        eye_distance_cm = 6.35

        eye1_1 = (shape1.part(36).x, shape1.part(36).x)
        eye1_2 = (shape1.part(42).x, shape1.part(42).x)

        nose1 = shape1.part(30).x
        nose2 = shape2.part(30).x
        distance_px = nose1 - nose2
        # 计算像素欧几里德距离
        eye_distance_px = np.linalg.norm(np.array(eye1_2) - np.array(eye1_1))
        px_to_meter = eye_distance_cm / eye_distance_px  # 像素转换
        distance = distance_px * px_to_meter

        # 确保返回距离为正数
        return abs(distance)

    # 检测交互
    def detect_interaction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度处理
        faces = self.detector(gray)  # 检测所有人脸
        recognized_faces = []  # 存储同一帧内识别出的人脸

        for face in faces:
            # 获取人脸的68个关键点并进行对人脸进行对齐
            # print('获取人脸')
            shape = self.predictor(gray, face)
            face_chip = dlib.get_face_chip(frame, shape, size=150)
            # 计算输入人脸的特征向量
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(face_chip))

            # KNN
            closest_distances = self.knn_clf.kneighbors([face_encoding], n_neighbors=1)
            is_recognized = closest_distances[0][0][0] <= 0.4

            label = "Unknown"
            if is_recognized:
                label = self.knn_clf.predict([face_encoding])[0]

            # print(f'{label}')
            recognized_faces.append((face, shape, label))

        interactions = []

        # 遍历同一帧内的所有人脸，计算它们之间的距离
        for i, (face1, shape1, label1) in enumerate(recognized_faces):
            for j, (face2, shape2, label2) in enumerate(recognized_faces):
                if i >= j:
                    continue
                print(f'{label1}')
                print(f'{label2}')
                # 判断身份是否符合
                if self.volunteer_label in label1 and self.elderly_label in label2 or self.volunteer_label in label2 and self.elderly_label in label1:
                    # distance = self.calculate_distance(face1, face2)
                    distance = self.calculate_distance(shape1, shape2)
                    print(f'{distance}cm')
                    # 1m内则存储交互信息
                    if distance <= self.distance_threshold:
                        print('检测到交互')
                        interactions.append((face1, face2, distance, label1, label2))

        return recognized_faces, interactions


def main():
    interaction_detector = InteractionDetection(
        knn_model_path=KNN_MODEL_PATH,
        predictor_path=PREDICTOR_PATH,
        face_rec_model_path=FACE_RECOGNITION_MODEL_PATH,
        distance_threshold=DISTANCE_THRESHOLD
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: 无法打开视频设备.")
        return

    frame_count = 0  # 帧计数器

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % SKIP_FRAMES == 0:
            recognized_faces, interactions = interaction_detector.detect_interaction(frame)

            # 绘制所有人脸框和标签
            for face, shape, label in recognized_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 绘制交互信息
            for face1, face2, distance, label1, label2 in interactions:
                x1, y1, w1, h1 = face1.left(), face1.top(), face1.width(), face1.height()
                x2, y2, w2, h2 = face2.left(), face2.top(), face2.width(), face2.height()

                # 在人脸上方显示标签和距离信息
                cv2.line(frame, (x1 + w1 // 2, y1 + h1 // 2), (x2 + w2 // 2, y2 + h2 // 2), (255, 0, 0), 2)
                cv2.putText(frame, f"{distance:.2f}cm", ((x1 + x2) // 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Interaction Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
