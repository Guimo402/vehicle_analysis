# 导入所需的库
import cv2  # OpenCV库用于视频处理和图像显示
from ultralytics import YOLO  # 导入YOLO目标检测模型
import os
import numpy as np
from collections import defaultdict
# 设置环境变量以解决潜在的库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载YOLO模型
# 使用预训练的yolo11m.pt模型文件初始化YOLO对象
model = YOLO("yolo11m.pt")

# 打开视频文件
video_path = "test.mp4"  # 设置视频文件路径
cap = cv2.VideoCapture(video_path)  # 创建视频捕获对象

# 获取视频的帧率,用于计算速度
fps = cap.get(cv2.CAP_PROP_FPS)

# 用于存储上一帧中目标的位置
prev_positions = defaultdict(lambda: None)
# 用于存储目标的速度
speeds = defaultdict(float)

# 获取视频图像尺寸,用于透视校正
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 当前车辆速度（固定为80 km/h）
current_vehicle_speed = 80.0  # km/h

# 速度计算参数
K_FACTOR = 200.0  # 增大速度缩放因子
SMOOTH_FACTOR = 0.5  # 减小平滑系数，使速度变化更快
DISTANCE_SCALE = 0.2  # 增大距离缩放因子

# 循环处理视频帧
while cap.isOpened():  # 当视频文件成功打开时循环
    # 读取一帧视频
    success, frame = cap.read()  # success表示是否读取成功,frame为读取到的帧

    if success:
        # 使用YOLO模型对帧进行目标跟踪
        # persist=True表示在帧之间保持跟踪状态
        results = model.track(frame, persist=True)
        
        # 获取跟踪结果
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()  # 获取边界框坐标(x,y,w,h)
            track_ids = results[0].boxes.id.cpu()  # 获取跟踪ID
            
            # 计算每个目标的速度
            for box, track_id in zip(boxes, track_ids):
                track_id = int(track_id)
                current_pos = box[:2].numpy()  # 获取当前位置(x,y)
                current_size = box[2:4].numpy()  # 获取当前目标的宽高
                
                if prev_positions[track_id] is not None:
                    # 计算像素距离
                    distance = np.linalg.norm(current_pos - prev_positions[track_id])
                    
                    # 透视校正系数计算
                    # 1. 基于y坐标的校正（远处的物体在图像上方）
                    y_factor = 1.0 + (1.0 - current_pos[1] / frame_height) * 2.0  # 增大校正系数
                    
                    # 2. 基于物体大小的校正（远处的物体在图像中显示较小）
                    # 计算物体在图像中所占的相对面积
                    relative_area = (current_size[0] * current_size[1]) / (frame_width * frame_height)
                    # 面积越小，校正系数越大
                    size_factor = 1.0 + (1.0 - min(relative_area * 1000, 0.9)) * 2.0  # 增大校正系数
                    
                    # 3. 基于x坐标的校正（考虑视角差异）
                    x_factor = 1.0 + 0.2 * abs(current_pos[0] / frame_width - 0.5)  # 增大视角差异影响
                    
                    # 4. 基于目标在图像中的位置进行额外校正（右下方的车辆）
                    position_factor = 1.0
                    if current_pos[0] > frame_width * 0.75 and current_pos[1] > frame_height * 0.75:
                        position_factor = 0.7  # 增大位置校正影响
                    
                    # 综合校正系数
                    correction_factor = y_factor * size_factor * x_factor * position_factor
                    
                    # 将像素距离转换为近似的实际距离
                    distance_meters = distance * DISTANCE_SCALE * correction_factor
                    
                    # 计算相对速度，增大速度变化
                    relative_speed = current_vehicle_speed * (1 - (distance_meters / 30))  # 减小分母使速度变化更大
                    if relative_speed < 0:
                        relative_speed = 0  # 速度不能为负
                    
                    # 转换为km/h并应用平滑
                    speeds[track_id] = speeds[track_id] * SMOOTH_FACTOR + (relative_speed) * (1 - SMOOTH_FACTOR)
                
                prev_positions[track_id] = current_pos

        # 在帧上可视化检测和跟踪结果
        annotated_frame = results[0].plot()
        
        # 在标注框上添加速度信息
        if results[0].boxes.id is not None:
            for box, track_id in zip(boxes, track_ids):
                track_id = int(track_id)
                if speeds[track_id] > 0:
                    x, y = int(box[0]), int(box[1])
                    speed_text = f"{speeds[track_id]:.1f} km/h"
                    cv2.putText(annotated_frame, speed_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示带有标注的帧
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # 检测键盘输入,如果按下'q'键则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果读取失败(到达视频末尾),退出循环
        break

# 释放资源
cap.release()  # 释放视频捕获对象
cv2.destroyAllWindows()  # 关闭所有显示窗口