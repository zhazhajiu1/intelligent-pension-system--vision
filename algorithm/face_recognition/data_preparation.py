import cv2
import os
import dlib
import imgaug.augmenters as iaa
import numpy as np
import random
from tqdm import tqdm

# 数据预处理
def preprocess_and_align_faces(input_dir='data',
                               predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat'):
    # 定义增强序列
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转50%的图像
        iaa.Affine(rotate=(-20, 20)),  # 随机旋转图像
        iaa.GaussianBlur(sigma=(0, 1.0)),  # 应用高斯模糊
        iaa.Multiply((0.8, 1.2)),  # 随机改变亮度
        iaa.LinearContrast((0.75, 1.5)),  # 随机改变对比度
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 添加高斯噪声
        iaa.Crop(percent=(0, 0.1))  # 随机裁剪图像
    ])

    # 加载 Dlib 的68个关键点检测器模型
    predictor = dlib.shape_predictor(predictor_path)
    # 加载 Dlib 的正向人脸检测器模型
    detector = dlib.get_frontal_face_detector()

    identities = os.listdir(input_dir)
    for category in tqdm(identities, desc="Processing identities"):
    # for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        if not os.path.isdir(category_dir):
            continue

        for person_name in os.listdir(category_dir):
            person_dir = os.path.join(category_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                if img_file.startswith("aligned_") or img_file.startswith("flipped_") or img_file.startswith("aug_"):
                    # 跳过处理有aligned_和flipped_和aug_前缀的照片
                    continue

                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像文件: {img_path}")
                    continue

                # 图像转为灰度图像
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用人脸检测器找到人脸
                faces = detector(gray)

                for face in faces:
                    shape = predictor(gray, face)
                    # 人脸对齐,大小调整为size×size像素224-150
                    aligned_face = dlib.get_face_chip(img, shape, size=224)
                    # 保存对齐后的图像，覆盖之前的文件
                    # cv2.imwrite(img_path, aligned_face)
                    aligned_img_path = os.path.join(person_dir, f"aligned_{img_file}")
                    cv2.imencode('.jpg', aligned_face)[1].tofile(aligned_img_path)

                    # 数据增强（例如，翻转图像）
                    flipped_face = cv2.flip(aligned_face, 1)
                    flipped_img_path = os.path.join(person_dir, f"flipped_{img_file}")
                    cv2.imencode('.jpg', flipped_face)[1].tofile(flipped_img_path)

                    # 应用图像增强
                    augmented_images = seq(images=[aligned_face])
                    for idx, aug_img in enumerate(augmented_images):
                        aug_img_path = os.path.join(person_dir, f"aug_{idx}_{img_file}")
                        cv2.imencode('.jpg', aug_img)[1].tofile(aug_img_path)

# 对齐人脸
preprocess_and_align_faces(input_dir='data', predictor_path='data_dlib/shape_predictor_68_face_landmarks.dat')

def preprocess_data(data_dir, predictor_path):
    X = []
    y = []

    # 人脸特征提取--ResNet
    face_rec_model = dlib.face_recognition_model_v1('data_dlib/dlib_face_recognition_resnet_model_v1.dat')
    # 加载 Dlib 的68个关键点检测器模型
    predictor = dlib.shape_predictor(predictor_path)
    # 加载 Dlib 的正向人脸检测器模型
    detector = dlib.get_frontal_face_detector()
    # 定义增强序列
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转50%的图像
        iaa.Affine(rotate=(-20, 20)),  # 随机旋转图像
        iaa.GaussianBlur(sigma=(0, 1.0)),  # 应用高斯模糊
        iaa.Multiply((0.8, 1.2)),  # 随机改变亮度
        iaa.LinearContrast((0.75, 1.5)),  # 随机改变对比度
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # 添加高斯噪声
        iaa.Crop(percent=(0, 0.1))  # 随机裁剪图像
    ])

    identities = os.listdir(data_dir)
    for identity in tqdm(identities, desc="Processing identities"):
    # for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)
        if not os.path.isdir(identity_dir):
            continue

        for person_name in os.listdir(identity_dir):
            person_dir = os.path.join(identity_dir, person_name)
            person_encodings = []
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                if img_file.startswith("aligned_") or img_file.startswith("flipped_") or img_file.startswith("aug_"):
                    # 跳过处理有aligned_和flipped_和aug_前缀的照片
                    continue

                img_path = os.path.join(person_dir, img_file)
                # img = cv2.imread(img_path)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"无法读取图像文件: {img_path}")
                    continue

                # 图像转为灰度图像
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用人脸检测器找到人脸
                faces = detector(gray)

                for face in faces:
                    shape = predictor(gray, face)
                    # 计算人脸特征向量--128维特征向量
                    encoding = np.array(face_rec_model.compute_face_descriptor(img, shape))
                    person_encodings.append(encoding)

                    # 人脸对齐,大小调整为size×size像素224-150
                    aligned_face = dlib.get_face_chip(img, shape, size=150)
                    aligned_encoding = np.array(face_rec_model.compute_face_descriptor(aligned_face))
                    person_encodings.append(aligned_encoding)
                    # 保存对齐后的图像，覆盖之前的文件
                    # cv2.imwrite(img_path, aligned_face)
                    # aligned_img_path = os.path.join(person_dir, f"aligned_{img_file}")
                    # cv2.imencode('.jpg', aligned_face)[1].tofile(aligned_img_path)

                    # 数据增强（翻转图像）
                    # flipped_face = cv2.flip(aligned_face, 1)
                    # encoding = np.array(face_rec_model.compute_face_descriptor(flipped_face))
                    # person_encodings.append(encoding)
                    # 数据增强：翻转图像并计算特征向量
                    flipped_face = cv2.flip(img, 1)
                    gray_flipped = cv2.cvtColor(flipped_face, cv2.COLOR_BGR2GRAY)
                    faces_flipped = detector(gray_flipped)

                    for face_flipped in faces_flipped:
                        shape_flipped = predictor(gray_flipped, face_flipped)
                        aligned_flipped_face = dlib.get_face_chip(flipped_face, shape_flipped, size=150)
                        flipped_encoding = np.array(face_rec_model.compute_face_descriptor(aligned_flipped_face))
                        person_encodings.append(flipped_encoding)
                    # flipped_img_path = os.path.join(person_dir, f"flipped_{img_file}")
                    # cv2.imencode('.jpg', flipped_face)[1].tofile(flipped_img_path)

                    # 应用图像增强
                    # augmented_images = seq(images=[aligned_face])
                    # # 生成人脸编码
                    # for aug_img in augmented_images:
                    #     aug_img_resized = cv2.resize(aug_img, (150, 150))
                    #     aug_encoding = np.array(face_rec_model.compute_face_descriptor(aug_img_resized))
                    #     person_encodings.append(aug_encoding)
                    # 保存图像
                    # for idx, aug_img in enumerate(augmented_images):
                    #     aug_img_path = os.path.join(person_dir, f"aug_{idx}_{img_file}")
                    #     cv2.imencode('.jpg', aug_img)[1].tofile(aug_img_path)

            if person_encodings:
                mean_encoding = np.mean(person_encodings, axis=0)
                X.append(mean_encoding)
                y.append(f"{identity}_{person_name}")

    return X, y