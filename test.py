import cv2  # OpenCV 库用于视频处理和图像显示
from ultralytics import YOLO  # 导入YOLO目标检测模型
import os
import numpy as np
from collections import defaultdict

# 设置环境变量以解决潜在的库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------
# 参数设置
# -------------------------------
VIDEO_PATH = "test.mp4"            # 视频文件路径
MODEL_PATH = "yolo11m.pt"            # YOLO 模型文件路径
CURRENT_VEHICLE_SPEED = 110.0         # 当前车辆速度（单位：km/h）
K_FACTOR = 200.0                   # 用于放大速度变化的缩放因子（可依据实际情况调整）
SMOOTH_FACTOR = 0.5                # 平滑因子（用于滤波平滑速度变化）
DISTANCE_SCALE = 0.5               # 像素距离转换到实际距离的基础因子（需依据标定调整）

# -------------------------------
# 定义工具函数
# -------------------------------
def compute_correction_factor(current_pos, current_size, frame_width, frame_height):
    """
    根据物体在图像中的位置和大小计算透视校正系数
    :param current_pos: 物体中心坐标 (x, y)
    :param current_size: 物体宽高 (w, h)
    :param frame_width: 图像宽度
    :param frame_height: 图像高度
    :return: 校正系数 (float)
    """
    # 基于 y 坐标的校正（图像上方的目标通常在远处，放大校正）
    y_factor = 1.0 + (1.0 - current_pos[1] / frame_height) * 2.0

    # 基于物体在图像中所占面积的校正：面积越小，校正因子越大
    relative_area = (current_size[0] * current_size[1]) / (frame_width * frame_height)
    size_factor = 1.0 + (1.0 - min(relative_area * 1000, 0.9)) * 2.0

    # 考虑 x 坐标的视角差异（距离左右图像中心越远，校正因子越大）
    x_factor = 1.0 + 0.2 * abs(current_pos[0] / frame_width - 0.5)

    # 针对特定区域（如右下角）的额外校正
    position_factor = 1.0
    if current_pos[0] > frame_width * 0.75 and current_pos[1] > frame_height * 0.75:
        position_factor = 0.7

    correction_factor = y_factor * size_factor * x_factor * position_factor
    return correction_factor

def estimate_speed(prev_pos, curr_pos, current_size, frame_width, frame_height):
    """
    根据两帧中目标位移及位置校正估计目标相对速度
    :param prev_pos: 上一帧目标中心坐标 (x, y)
    :param curr_pos: 当前帧目标中心坐标 (x, y)
    :param current_size: 当前目标宽高 (w, h)
    :param frame_width: 图像宽度
    :param frame_height: 图像高度
    :return: 估计速度（km/h）
    """
    # 计算像素位移
    pixel_distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    # 获取透视校正系数（根据当前目标位置和大小计算）
    corr_factor = compute_correction_factor(curr_pos, current_size, frame_width, frame_height)
    # 将像素位移转换为实际距离，加入校正因子
    distance_meters = pixel_distance * DISTANCE_SCALE * corr_factor
    # 根据距离与一个经验阈值（例如 30 米，可按实际场景调整）计算目标相对速度
    # 这里使用 K_FACTOR 来增强速度变化的敏感度
    relative_speed = CURRENT_VEHICLE_SPEED * (1 - (distance_meters / K_FACTOR))
    # 避免负速度
    if relative_speed < 0:
        relative_speed = 0.0
    return relative_speed

# -------------------------------
# 模型和视频初始化
# -------------------------------
model = YOLO(MODEL_PATH)  # 加载预训练的 YOLO 模型

cap = cv2.VideoCapture(VIDEO_PATH)  # 打开视频文件
if not cap.isOpened():
    raise IOError("无法打开视频文件: " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 获取视频帧尺寸（用于透视校正的计算）
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 用于存储每个跟踪目标上一帧的位置以及平滑后的速度
prev_positions = defaultdict(lambda: None)
speeds = defaultdict(float)

# -------------------------------
# 主循环：处理每一帧、跟踪目标、计算并显示速度
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型对帧进行目标跟踪，保持跟踪状态以便跨帧关联
    results = model.track(frame, persist=True)

    # 获取跟踪结果（判断当前帧是否存在有效跟踪目标）
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()  # 边界框格式：x,y,w,h
        track_ids = results[0].boxes.id.cpu()  # 跟踪目标 ID

        # 对每个检测目标进行处理
        for box, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            # 当前目标中心点
            current_pos = box[:2].numpy()
            # 当前目标尺寸
            current_size = box[2:4].numpy()

            # 如果在上一帧中存在该目标，计算速度
            if prev_positions[track_id] is not None:
                # 估计相对速度（单位 km/h），并加入一定平滑
                curr_speed = estimate_speed(prev_positions[track_id], current_pos, current_size, frame_width, frame_height)
                speeds[track_id] = speeds[track_id] * SMOOTH_FACTOR + curr_speed * (1 - SMOOTH_FACTOR)
            # 更新该目标的位置
            prev_positions[track_id] = current_pos

    # 获取可视化跟踪结果
    annotated_frame = results[0].plot()

    # 在跟踪框上叠加速度信息
    if results[0].boxes.id is not None:
        for box, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            if speeds[track_id] > 0:
                x, y = int(box[0]), int(box[1])
                speed_text = f"{speeds[track_id]:.1f} km/h"
                cv2.putText(annotated_frame, speed_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLO11 Tracking - 优化版", annotated_frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放视频资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
