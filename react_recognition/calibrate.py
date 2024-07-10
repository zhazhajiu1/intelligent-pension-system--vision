import cv2
import dlib
import numpy as np
import pickle
from sklearn import neighbors
import imutils
from imutils import paths, face_utils

PREDICTOR_PATH = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = '../face_recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

# 使用视频流实时检测人脸/A4纸并计算焦距
def calibrate_using_video(knownDistance, knownWidth):
    cap = cv2.VideoCapture(0)
    focalLength = None

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # a4纸
        # marker = find_marker(frame)
        # if marker is not None:
        #     focalLength = (marker[1][0] * knownDistance) / knownWidth
        #     print(f'A4实际距离: {knownDistance}cm')
        #     print(f'A4实际宽度: {knownWidth}cm')
        #     print(f'A4像素: {marker[1][0]}px')
        #     print(f'焦距: {focalLength}px')
        #
        #     # 画出检测到的矩形框
        #     box = cv2.boxPoints(marker)
        #     box = np.int0(box)
        #     cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        #     cv2.putText(frame, f'Focal Length: {focalLength:.2f}px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #                 (0, 255, 0), 2)

        # 人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[0]
            right_eye = shape[16]
            face_width_px = np.linalg.norm(left_eye - right_eye)

            focalLength = (face_width_px * knownDistance) / knownWidth
            print(f'实际距离: {knownDistance}cm')
            print(f'实际宽度: {knownWidth}cm')
            print(f'人脸宽度: {face_width_px}px')
            print(f'焦距: {focalLength}px')

            # 画出检测到的人脸
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, f'Focal Length: {focalLength:.2f}px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

        cv2.imshow('Calibration', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return focalLength

def find_marker(image):
    # 在图像中寻找目标物体的轮廓
    # 将图像转换为灰度图，然后进行模糊处理，以去除图像中的高频噪声
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用 Canny 算法进行边缘检测
    edged = cv2.Canny(gray, 35, 125)
    # 寻找边缘图像中的轮廓，保留最大的一个，假设这是我们图像中的纸张
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None
    c = max(cnts, key=cv2.contourArea)
    # 计算纸张区域的边界框，并返回
    return cv2.minAreaRect(c)

def main():
    knownDistance = 30.0  # A4纸与摄像头的实际距离，单位：厘米
    knownWidth = 15.0  # 实际宽度，单位：厘米

    focalLength = calibrate_using_video(knownDistance, knownWidth)
    print(f'最终计算的焦距: {focalLength}px')

if __name__ == '__main__':
    main()