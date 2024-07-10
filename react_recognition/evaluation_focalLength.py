import cv2
import dlib
import numpy as np
from imutils import face_utils

KNOWN_FOCAL_LENGTH = 630.0  # 焦距px
KNOWN_WIDTH = 15.0  # 人脸的实际宽度cm
PREDICTOR_PATH = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = '../face_recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

# 实时检测人脸并计算实际宽度和距离
def measure_face_width_and_distance(frame, predictor_path):
    eye_distance_cm = 7
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36]
        right_eye = shape[42]
        left_face = shape[0]
        right_face = shape[16]
        eye_width_px = np.linalg.norm(left_eye - right_eye)
        face_width_px = np.linalg.norm(left_face - right_face)
        face_width_cm = eye_distance_cm / eye_width_px * face_width_px

        distance = face_width_cm * KNOWN_FOCAL_LENGTH / face_width_px

        return face_width_cm, distance
    else:
        return None, None


def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    predictor_path = PREDICTOR_PATH

    if not cap.isOpened():
        print("Error: 无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: 无法捕获图像")
            break

        face_width_cm, distance = measure_face_width_and_distance(frame, predictor_path)

        if face_width_cm is not None:
            cv2.putText(frame, f"Width: {face_width_cm:.2f}cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Distance Measurement', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
