# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                             QSpinBox, QGroupBox, QDoubleSpinBox, QStatusBar,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from collections import defaultdict, deque
import os
import time # 需要 time 模块

# 设置环境变量 (可选，用于解决某些环境下的库冲突问题)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class ObjectTrackingUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 初始化速度估计参数 ---
        self.CAMERA_SPEED_KMH = 80.0
        self.PERSPECTIVE_POWER = 1.5
        self.K_FACTOR = 50.0
        self.MIN_HISTORY_FOR_SPEED = 10
        self.REFERENCE_Y = None # 将在获取帧尺寸后设置
        self.DELTA_T = 1.0 / 30.0 # 默认帧时间间隔，将在加载视频后更新
        self.FRAME_H = None # 帧高度
        self.FRAME_W = None # 帧宽度

        # --- 跟踪状态变量 ---
        self.track_history = defaultdict(lambda: deque(maxlen=60)) # 使用 deque
        self.estimated_speeds = {}

        # --- 初始化其他变量 ---
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_processing = False
        self.current_frame_count = 0 # 用于计算时间戳

        self.initUI()

    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('车辆跟踪与速度估计系统')
        self.setGeometry(100, 100, 1400, 900) # 稍微增大窗口尺寸

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10) # 设置外边距
        layout.setSpacing(10) # 设置控件间距

        # --- 左侧控制面板 ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(350) # 限制控制面板宽度
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(15) # 控制面板内间距

        # --- 模型加载组 ---
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout() # 改为垂直布局更好看
        self.model_path_label = QLabel('未加载模型')
        self.model_path_label.setWordWrap(True) # 允许换行
        self.load_model_btn = QPushButton('加载YOLO模型')
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)
        model_group.setLayout(model_layout)

        # --- 视频选择组 ---
        video_group = QGroupBox("视频设置")
        video_layout = QVBoxLayout() # 改为垂直布局
        self.video_path_label = QLabel('未选择视频')
        self.video_path_label.setWordWrap(True)
        self.load_video_btn = QPushButton('选择视频文件')
        self.load_video_btn.clicked.connect(self.load_video)
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.load_video_btn)
        video_group.setLayout(video_layout)

        # --- 参数设置组 ---
        params_group = QGroupBox("参数调整")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(10)

        # 基准速度设置
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel('基准速度 (km/h):'))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(0, 200)
        self.speed_spinbox.setValue(int(self.CAMERA_SPEED_KMH))
        self.speed_spinbox.valueChanged.connect(self.update_base_speed)
        speed_layout.addWidget(self.speed_spinbox)
        params_layout.addLayout(speed_layout)

        # 置信度阈值设置
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel('置信度阈值:'))
        self.conf_spinbox = QDoubleSpinBox() # 改为 DoubleSpinBox
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.3) # 默认值改为0.3
        conf_layout.addWidget(self.conf_spinbox)
        params_layout.addLayout(conf_layout)

        # K_FACTOR 设置
        k_factor_layout = QHBoxLayout()
        k_factor_layout.addWidget(QLabel('K因子 (速度缩放):'))
        self.k_factor_spinbox = QDoubleSpinBox()
        self.k_factor_spinbox.setRange(1.0, 1000.0) # 范围扩大
        self.k_factor_spinbox.setSingleStep(10.0)
        self.k_factor_spinbox.setValue(self.K_FACTOR)
        self.k_factor_spinbox.valueChanged.connect(self.update_k_factor)
        k_factor_layout.addWidget(self.k_factor_spinbox)
        params_layout.addLayout(k_factor_layout)

        # PERSPECTIVE_POWER 设置
        pers_power_layout = QHBoxLayout()
        pers_power_layout.addWidget(QLabel('透视补偿指数:'))
        self.pers_power_spinbox = QDoubleSpinBox()
        self.pers_power_spinbox.setRange(0.5, 3.0)
        self.pers_power_spinbox.setSingleStep(0.1)
        self.pers_power_spinbox.setValue(self.PERSPECTIVE_POWER)
        self.pers_power_spinbox.valueChanged.connect(self.update_pers_power)
        pers_power_layout.addWidget(self.pers_power_spinbox)
        params_layout.addLayout(pers_power_layout)

        # MIN_HISTORY_FOR_SPEED 设置
        min_hist_layout = QHBoxLayout()
        min_hist_layout.addWidget(QLabel('速度计算历史帧数:'))
        self.min_hist_spinbox = QSpinBox()
        self.min_hist_spinbox.setRange(3, 60)  # 最小3帧，最大60帧
        self.min_hist_spinbox.setValue(self.MIN_HISTORY_FOR_SPEED)
        self.min_hist_spinbox.valueChanged.connect(self.update_min_hist)
        min_hist_layout.addWidget(self.min_hist_spinbox)
        params_layout.addLayout(min_hist_layout)


        # --- 控制按钮 ---
        self.start_btn = QPushButton('开始处理')
        self.start_btn.clicked.connect(self.toggle_processing)
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(40) # 增大按钮高度

        # --- 将所有组件添加到控制面板布局 ---
        control_layout.addWidget(model_group)
        control_layout.addWidget(video_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.start_btn)
        control_layout.addStretch() # 添加伸缩因子，将控件推到顶部

        # --- 右侧视频显示 ---
        self.video_label = QLabel("请先加载模型和视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        # 设置视频标签可以缩放
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- 将控制面板和视频显示添加到主布局 ---
        layout.addWidget(control_panel)
        layout.addWidget(self.video_label, 1) # 第二个参数为拉伸因子

        # --- 状态栏 ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统准备就绪")

    def load_model(self):
        """加载YOLO模型"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PT文件 (*.pt)")
        if file_name:
            try:
                self.model = YOLO(file_name)
                # 尝试进行一次推理以确保模型可用
                _ = self.model(np.zeros((640, 640, 3), dtype=np.uint8))
                self.model_path_label.setText(f"已加载: {os.path.basename(file_name)}")
                self.statusBar.showMessage(f"模型 '{os.path.basename(file_name)}' 加载成功", 5000)
                self.check_start_conditions()
            except Exception as e:
                self.model_path_label.setText(f'模型加载失败')
                self.statusBar.showMessage(f"模型加载失败: {e}", 5000)
                self.model = None # 确保模型置空

    def load_video(self):
        """加载视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            # 如果正在处理，先停止
            if self.is_processing:
                self.toggle_processing()

            # 释放旧的视频捕获
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                # 获取视频属性
                self.FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.DELTA_T = 1.0 / fps
                else:
                    self.DELTA_T = 1.0 / 30.0 # 默认值
                    self.statusBar.showMessage("警告: 无法获取视频FPS，假设为30", 3000)

                # 计算 REFERENCE_Y
                self.REFERENCE_Y = self.FRAME_H / 4.0

                self.video_path_label.setText(f"已加载: {os.path.basename(file_name)}")
                self.statusBar.showMessage(f"视频 '{os.path.basename(file_name)}' 加载成功", 5000)
                self.check_start_conditions()
                # 显示第一帧作为预览
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置到视频开头

            else:
                self.video_path_label.setText('视频加载失败')
                self.statusBar.showMessage("视频加载失败", 5000)
                self.cap = None

    # --- 参数更新槽函数 ---
    def update_base_speed(self, value):
        self.CAMERA_SPEED_KMH = float(value)

    def update_k_factor(self, value):
        self.K_FACTOR = value

    def update_pers_power(self, value):
        self.PERSPECTIVE_POWER = value

    def update_min_hist(self, value):
        self.MIN_HISTORY_FOR_SPEED = value

    def check_start_conditions(self):
        """检查是否满足开始处理的条件"""
        ready = self.model is not None and self.cap is not None
        self.start_btn.setEnabled(ready)
        if not ready:
             self.start_btn.setText('开始处理')
             self.is_processing = False
             self.timer.stop()


    def toggle_processing(self):
        """切换处理状态"""
        if not self.is_processing:
            if self.cap is None or self.model is None:
                self.statusBar.showMessage("错误: 请先加载模型和视频", 5000)
                return

            # 重置跟踪历史和速度
            self.track_history.clear()
            self.estimated_speeds.clear()
            self.current_frame_count = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 确保从头开始

            self.is_processing = True
            self.start_btn.setText('停止处理')
            self.statusBar.showMessage("开始处理视频...")
            # 根据视频的FPS设置定时器间隔可能更准确，但简单起见先用固定值
            timer_interval = int(self.DELTA_T * 1000) if self.DELTA_T > 0 else 33 # 约30fps
            self.timer.start(max(1, timer_interval)) # 至少1ms间隔
        else:
            self.is_processing = False
            self.start_btn.setText('开始处理')
            self.timer.stop()
            self.statusBar.showMessage("处理已停止", 5000)

    def update_frame(self):
        """更新视频帧并进行处理"""
        if self.cap is None or not self.cap.isOpened() or not self.is_processing:
            self.toggle_processing() # 如果发生错误则停止
            return

        ret, frame = self.cap.read()
        if not ret:
            # 视频结束，停止处理或循环播放
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 循环播放
            # self.current_frame_count = 0
            # self.track_history.clear()
            # self.estimated_speeds.clear()
            # return
            self.toggle_processing() # 视频结束则停止
            self.statusBar.showMessage("视频处理完成", 5000)
            return

        # 确保 REFERENCE_Y 已设置
        if self.REFERENCE_Y is None and self.FRAME_H is not None:
             self.REFERENCE_Y = self.FRAME_H / 2.0

        current_time_sec = self.current_frame_count * self.DELTA_T
        processed_frame = frame.copy() # 在副本上绘制

        # --- 目标跟踪处理 ---
        try:
            results = self.model.track(
                frame, # 使用原始帧进行推理
                persist=True,
                conf=self.conf_spinbox.value(),
                tracker="bytetrack.yaml", # 或者其他 tracker
                classes=[2, 3, 5, 7], # 只跟踪车辆相关类别
                verbose=False # 关闭冗余输出
            )

            # 获取跟踪结果
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                # clss = results[0].boxes.cls.cpu().numpy().astype(int) # 如果需要类别信息

                current_tracked_ids = set()

                # --- 遍历当前帧的跟踪结果 ---
                for box, track_id in zip(boxes_xyxy, track_ids):
                    current_tracked_ids.add(track_id)
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = y2 # 使用底部坐标

                    # 更新跟踪历史
                    self.track_history[track_id].append((current_time_sec, cx, cy))

                    # --- 速度估计 ---
                    if len(self.track_history[track_id]) >= self.MIN_HISTORY_FOR_SPEED:
                        try:
                            points = list(self.track_history[track_id])[-self.MIN_HISTORY_FOR_SPEED:]
                            times = np.array([p[0] for p in points])
                            ys = np.array([p[2] for p in points]) # cy is y2

                            if len(times) > 1 and (times[-1] - times[0]) > 1e-3:
                                times_rel = times - times[0]
                                coeffs = np.polyfit(times_rel, ys, 1)
                                vy_pixel_per_sec = coeffs[0]

                                cy_curr = ys[-1]
                                cy_curr_clamped = max(1.0, float(cy_curr))

                                # 透视补偿
                                if self.REFERENCE_Y is not None: # 确保 REFERENCE_Y 已计算
                                    perspective_scale = (self.REFERENCE_Y / cy_curr_clamped) ** self.PERSPECTIVE_POWER
                                    perspective_scale = np.clip(perspective_scale, 0.1, 10.0)
                                else:
                                    perspective_scale = 1.0 # REFERENCE_Y 未就绪则不缩放

                                # 相对速度
                                relative_speed_kmh = -self.K_FACTOR * vy_pixel_per_sec * perspective_scale

                                # 最终速度
                                estimated_speed_kmh = self.CAMERA_SPEED_KMH + relative_speed_kmh
                                estimated_speed_kmh = max(0, min(200, estimated_speed_kmh)) # 限制范围

                                self.estimated_speeds[track_id] = estimated_speed_kmh

                            else:
                                self.estimated_speeds.pop(track_id, None) # 时间跨度不足

                        except Exception as e:
                            # print(f"计算速度时出错 (ID: {track_id}): {e}") # 调试时取消注释
                            self.estimated_speeds.pop(track_id, None)
                    else:
                        self.estimated_speeds.pop(track_id, None) # 历史不足

                    # --- 手动绘制 ---
                    # 绘制边界框
                    color = (255, 150, 0) # 橙色
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                    # 准备显示文本 (ID 和 速度)
                    label = f"ID:{track_id}"
                    if track_id in self.estimated_speeds:
                        label += f" {self.estimated_speeds[track_id]:.1f}km/h"
                    else:
                        label += " ...km/h"

                    # 计算文本大小以绘制背景
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # 绘制文本背景 (放在框内顶部)
                    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
                    cv2.rectangle(processed_frame, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), (0,0,0), -1) # 黑色背景
                    # 绘制文本
                    cv2.putText(processed_frame, label, (x1, label_y - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # 绿色字体

                # --- 清理消失的ID ---
                disappeared_ids = set(self.estimated_speeds.keys()) - current_tracked_ids
                for gone_id in disappeared_ids:
                    self.estimated_speeds.pop(gone_id, None)
                    self.track_history.pop(gone_id, None)

            # 如果没有检测到物体，也显示原始帧
            self.display_frame(processed_frame)

        except Exception as e:
            print(f"处理帧 {self.current_frame_count} 时出错: {e}")
            self.display_frame(frame) # 出错时显示原始帧
            # 考虑在这里停止处理 self.toggle_processing()

        self.current_frame_count += 1


    def display_frame(self, frame):
        """将OpenCV帧转换为QPixmap并显示在QLabel上"""
        if frame is None:
            return
        try:
            # 转换颜色空间
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            if h == 0 or w == 0: 
                return
            
            # 获取显示区域的大小
            label_size = self.video_label.size()
            
            # 设置最大显示尺寸（可选）
            MAX_DISPLAY_WIDTH = 1280
            MAX_DISPLAY_HEIGHT = 720
            
            # 计算缩放比例
            w_ratio = min(label_size.width(), MAX_DISPLAY_WIDTH) / w
            h_ratio = min(label_size.height(), MAX_DISPLAY_HEIGHT) / h
            scale_ratio = min(w_ratio, h_ratio)
            
            # 计算新的尺寸
            new_width = int(w * scale_ratio)
            new_height = int(h * scale_ratio)
            
            # 创建并缩放图像
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                new_width, 
                new_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 设置显示
            self.video_label.setPixmap(scaled_pixmap)
            
            # 保存当前帧和尺寸信息（用于窗口大小调整）
            self.current_frame = frame
            self.current_frame_size = (new_width, new_height)
            
        except Exception as e:
            print(f"显示帧时出错: {e}")


    def closeEvent(self, event):
        """关闭窗口时的清理工作"""
        self.is_processing = False # 确保停止处理标志被设置
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        print("应用程序关闭，资源已释放。")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ObjectTrackingUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
# ```

# **主要改进说明:**

# 1.  **速度估计算法集成:** `update_frame` 中现在包含了之前脚本的核心速度计算逻辑，包括使用 `y2` 坐标、线性回归、透视补偿 (`perspective_scale`) 和 `K_FACTOR`。
# 2.  **GUI布局与样式:**
#     * 使用了 `QGroupBox` 将模型、视频和参数设置分组。
#     * 调整了布局的边距 (`setContentsMargins`) 和间距 (`setSpacing`)。
#     * 增加了 `QStatusBar` 用于显示操作反馈和状态信息。
#     * 视频显示区域 (`video_label`) 设置了背景色和边框，并允许其随窗口缩放。
#     * 控制面板设置了最大宽度，防止过宽。
# 3.  **参数控件:**
#     * 添加了 `QDoubleSpinBox` 用于调整 `K_FACTOR` 和 `PERSPECTIVE_POWER`。
#     * 添加了 `QSpinBox` 用于调整 `MIN_HISTORY_FOR_SPEED`。
#     * 置信度阈值也改为了 `QDoubleSpinBox`。
#     * 这些控件的值会实时更新类中的相应参数。
# 4.  **手动绘图:** 不再使用 `results[0].plot()`，而是通过 `cv2.rectangle` 和 `cv2.putText` 手动在 `processed_frame` 上绘制边界框和计算出的速度信息，提供了更大的灵活性。
# 5.  **状态管理与反馈:**
#     * `check_start_conditions` 函数用于控制 "开始处理" 按钮的可用性。
#     * `toggle_processing` 函数管理处理状态，并在状态栏显示相应信息。
#     * 加载模型和视频时会更新状态栏。
#     * 处理视频帧时增加了 `try...except` 块来捕获潜在错误。
# 6.  **变量初始化与清理:**
#     * 相关的参数和状态变量（如 `track_history`, `estimated_speeds`）在 `__init__` 中初始化。
#     * 在开始处理时会清空历史记录 (`toggle_processing`)。
#     * 在 `closeEvent` 中确保定时器停止和视频资源释放。
# 7.  **动态 `REFERENCE_Y`:** `REFERENCE_Y` 在视频成功加载后根据帧高度动态计算。

# **如何使用和调试:**

# 1.  **运行代码:** 确保安装了必要的库 (`PyQt5`, `opencv-python`, `ultralytics`, `numpy`)。
# 2.  **加载模型和视频:** 通过界面按钮选择。
# 3.  **调整参数:** 在界面右侧调整基准速度、置信度、K因子、透视补偿指数和历史帧数。**`K_FACTOR` 仍然是最需要根据您的视频进行调试的关键参数。**
# 4.  **开始处理:** 点击 "开始处理" 按钮。
# 5.  **观察结果:** 观察视频中车辆框上显示的速度。根据观察结果反复调整参数，直到满意为止。
# 6.  **状态栏信息:** 注意窗口底部的状态栏会提供操作反馈。

# 这个版本结合了两种方法的优点，并提供了一个功能更完善、更易于调整参数的用户