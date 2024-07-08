import cv2
import numpy as np
from detect import Detector  # 调用detect文件的Detector类
import tracker  # 调用修改的tracker文件

point1 = []  # 定义空列表，用于获取鼠标绘制的坐标
point2 = []


def draw_mask(im, point1, point2):  # 定义一个函数，用于将鼠标绘制的区域绘制到显示的图片上
    pts = np.array([[point1[0], point1[1]],
                    [point2[0], point1[1]],
                    [point2[0], point2[1]],
                    [point1[0], point2[1]]], np.int32)
    cv2.polylines(im, [pts], True, (255, 255, 0), 3)
    return im


def draw_rectangle(event, x, y, flags, param):  # 定义鼠标事件函数
    global im, point1, point2

    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(im, point1, (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cv2.rectangle(im, point1, point2, (0, 255, 0), 2)


cv2.namedWindow('main')  # 窗口名字为展示时窗口名
cv2.setMouseCallback('main', draw_rectangle)  # 在此窗口名中调用鼠标事件

if __name__ == '__main__':

    video = cv2.VideoCapture(0)  # Open default camera device
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame = video.read()
    frame_cnt = 0  # 记录帧数  用于判断帧数是否大于2，进行目标跟踪
    dict_box = dict()
    dic_id = dict()
    det = Detector()  # 生成预测对象
    print("run")
    while True:
        _, im = video.read()
        if im is None:
            print("out")
            break
        # im = cv2.resize(im,(960,640))
        frame_cnt += 1
        listbox = []  # box框

        im, bboxes = det.detect(im)  # 获取预测结果

        mask = np.zeros((height, width, 3), np.uint8)  # 掩膜区域数值设置为0
        if point1 != [] and point2 != []:  # 判断是否获取到鼠标事件的点
            cv2.rectangle(mask, (point1[0], point1[1]), (point2[0], point2[1]), 255, -1)  # 绘制区域，用作判断入

        if len(bboxes) > 0:  # 判断是否有预测值
            listboxs = tracker.update(bboxes, im, frame_cnt, dict_box, dic_id)  # 将预测值送入目标跟踪中
            im = tracker.draw_bboxes(im, listboxs, mask)  # 绘制在原图上
            if point1 != [] and point2 != []:  # 将既定区域绘制到输出图片上
                im = draw_mask(im, point1, point2)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出播放
            break

        cv2.imshow('main', im)
        cv2.waitKey(30)

    video.release()
    cv2.destroyAllWindows()


