import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from pathlib import Path
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

# 参数配置
weights = 'runs/detect/best.pt'  # 模型权重路径
save_path = r"runs/detect/webcam.mp4"  # 视频保存路径
img_size = 640  # 图像大小
stride = 32  # 步长77
half = False  # 是否使用半精度浮点数减少内存占用，需要GPU支持

# 导入YOLOv5模型
model = attempt_load(weights)

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示打开本地摄像头
frame = 0  # 开始处理的帧数

# 获取视频帧率、宽度和高度，设置输出视频的帧率和大小
ret_val, img0 = cap.read()
fps, w, h = 30, img0.shape[1], img0.shape[0]
vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# 计时器和状态标记
fall_timer = 0  # 摔倒计时器
fall_detected = False  # 摔倒检测标志

def scale_bboxes(det, im_shape, img0_shape):
    height, width = im_shape
    img0_height, img0_width, _ = img0_shape

    # Scale the bounding box coordinates
    det[:, 0] *= (img0_width / width)  # scale width
    det[:, 1] *= (img0_height / height)  # scale height
    det[:, 2] *= (img0_width / width)  # scale width
    det[:, 3] *= (img0_height / height)  # scale height

    return det

try:
    # 持续处理视频帧直到退出循环
    while True:
        ret_val, img0 = cap.read()  # 读取视频帧
        if not ret_val:
            break  # 如果没有读取到帧，则退出循环

        frame += 1  # 帧数自增
        print(f'Processing frame {frame}')

        # 对图像进行Padded resize
        img = letterbox(img0, img_size, stride=stride, auto=True)[0]

        # 转换图像格式
        img = img.transpose((2, 0, 1))[::-1]  # HWC转为CHW，BGR转为RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0  # 像素值归一化到[0.0, 1.0]
        img = img[None]  # [h w c] -> [1 h w c]

        # 模型推理
        pred = model(img)[0]  # 获取模型输出
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)  # 进行非最大抑制

        # 绘制边框和标签
        det = pred[0] if len(pred) else []  # 检测结果
        annotator = Annotator(img0.copy(), line_width=3, example=str(model.names))

        fall_detected = False  # 每帧默认未检测到摔倒

        if len(det):
            det = scale_bboxes(det, (img.shape[2], img.shape[3]), img0.shape)
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # 类别索引
                label = f'Fall ({conf:.2f})' if model.names[c] == 'fall' else f'nofall ({conf:.2f})'
                color = (0, 0, 255) if model.names[c] == 'fall' else (0, 255, 0)
                annotator.box_label(xyxy, label, color=color)  # 绘制边框和标签
                if model.names[c] == 'fall':  # 检查是否为'fall'状态
                    fall_detected = True
                    break  # 一旦检测到摔倒状态就跳出循环

        if fall_detected:
            fall_timer += 1  # 增加摔倒计时器
            print(fall_timer)
            if fall_timer >= 1 * fps:  # 如果摔倒超过2秒（fps=30，每秒30帧）
                print("摔倒")
                cv2.putText(img0, "老人摔倒啦！！", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite('fall_detected.jpg', img0)  # 保存截图
                fall_timer = 0  # 重置计时器
        else:
            fall_timer = 0  # 重置摔倒计时器

        # 写入视频帧
        im0 = annotator.result()
        vid_writer.write(im0)

        cv2.imshow('image', im0)  # 显示图像
        if cv2.waitKey(1) == ord('q'):
            # 按'q'键退出循环
            break
finally:
    # 释放视频写入对象和摄像头对象
    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f'Webcam finish, save to {save_path}')  # 显示处理完成的消息和视频保存路径