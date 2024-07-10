import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import argparse
from collections import deque

from model.initialization import initialization
from model.utils import evaluation
from config import conf

def load_image_from_camera(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def load_image_from_file(file_path):
    return Image.open(file_path)

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def process_image(model, img):
    img = F.to_tensor(img)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img = img.to(device)
    with torch.no_grad():
        prediction = model([img])

    if len(prediction[0]['masks']) == 0:
        return np.zeros(img.shape[1:], dtype=np.uint8)

    masks = prediction[0]['masks']
    masks = masks[0, 0].mul(255).byte().cpu().numpy()
    return masks

def main():
    m = initialization(conf)[0]
    print('Loading the model ...')
    m.load('GaitSet_CASIA-B_4_False_256_0.2_32_full_30-56000-encoder.ptm',
           'GaitSet_CASIA-B_4_False_256_0.2_32_full_30-56000-optimizer.ptm')

    mask_rcnn_model = get_instance_segmentation_model(num_classes=2)
    state_dict = torch.load("model.pth", map_location=torch.device('cpu'))
    mask_rcnn_model.load_state_dict(state_dict)
    mask_rcnn_model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        return

    frame_count = 0
    labels = deque(maxlen=30)
    most_common_label = ""
    display_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1
        image = load_image_from_camera(frame)

        if frame_count % 30 == 0:
            mask = process_image(mask_rcnn_model, image)
        else:
            mask = np.zeros_like(frame[:, :, 0])  # 用于跳过Mask R-CNN处理的占位符

        masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        label = m.predict_person(masked_image)
        labels.append(label)

        # cv2.putText(frame, f'Person: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if frame_count >= 30:
            most_common_label = max(set(labels), key=labels.count)
            frame_count = 0
            display_counter = 30  # 设置显示计数器

        # 如果显示计数器大于0，显示 most_common_label
        if display_counter > 0:
            cv2.putText(frame, f'Most common: {most_common_label}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            display_counter -= 1  # 递减显示计数器

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()