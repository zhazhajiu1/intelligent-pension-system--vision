import cv2
import os
import pyttsx3
import dlib
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# 初始化语音引擎
engine = pyttsx3.init()

# 初始化Dlib的检测器和预测器
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'  # 请替换为实际路径
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 设置语音提示
def speak(text):
    engine.say(text)
    engine.runAndWait()

# 中文正常显示
def draw_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    # 将OpenCV的图像格式转换为PIL的图像格式
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # 使用本地的中文字体
    font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    draw.text(position, text, font=font, fill=color)
    # 将PIL的图像格式转换为OpenCV的图像格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 检查人脸数量
def check_faces(frame, faces):
    if len(faces) == 0:
        speak("没有检测到人脸")
        frame = draw_chinese_text(frame, "没有检测到人脸", (10, 30))
        cv2.imshow("Collecting Faces", frame)
        cv2.waitKey(1500)
        return False
    elif len(faces) > 1:
        speak("发现多张人脸")
        frame = draw_chinese_text(frame, "发现多张人脸", (10, 30))
        cv2.imshow("Collecting Faces", frame)
        cv2.waitKey(1500)
        return False
    else:
        speak("可以开始采集图像了")
        frame = draw_chinese_text(frame, "可以开始采集图像了", (10, 30))
        cv2.imshow("Collecting Faces", frame)
        cv2.waitKey(2000)
        return True

# 保存图片
def save_images(cap, action, action_name, name, count=15):
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = draw_chinese_text(frame, action_name, (10, 30))
        cv2.imshow("Collecting Faces", frame)
        img_path = os.path.join(name, f"{action}_{name}_{i+1}.jpg")
        cv2.imwrite(img_path, frame)
        time.sleep(0.5)

# 采集人脸图像
def collect_faces(name):
    if not os.path.exists(name):
        os.makedirs(name)

    cap = cv2.VideoCapture(0)
    actions = {
        'blink': '请眨眼',
        'open_mouth': '请张嘴',
        'smile': '请笑一笑',
        'rise_head': '请抬头',
        'bow_head': '请低头',
        'look_left': '请看左边',
        'look_right': '请看右边'
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if check_faces(frame, faces):
            for action, action_name in actions.items():
                speak(f"1) {action_name}")
                save_images(cap, action, action_name, name)
                time.sleep(2)
            speak("采集完毕")
            break

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input("请输入姓名: ")
    collect_faces(name)
