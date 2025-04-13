import cv2
from ultralytics import YOLO
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载YOLOv8模型（请确保模型文件存在，如 yolov8n.pt）
model = YOLO("yolo11m.pt")

# 打开视频文件，如果需要使用摄像头，将参数改为0
cap = cv2.VideoCapture("test.mp4")

# 获取原始视频尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置显示窗口的缩放比例
scale_percent = 60  # 缩小到原来的60%

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8内置的tracker对当前帧进行跟踪
    # persist=True表示在连续帧中保持跟踪状态
    results = model.track(frame, persist=True)

    # 获取带有标注框的图像
    annotated_frame = results[0].plot()
    
    # 调整图像大小，使显示窗口变小
    width = int(frame_width * scale_percent / 100)
    height = int(frame_height * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(annotated_frame, dim, interpolation=cv2.INTER_AREA)

    # 显示处理后的帧（缩小版）
    cv2.imshow("YOLOv8 Tracking", resized_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
