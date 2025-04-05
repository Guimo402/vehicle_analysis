import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from collections import defaultdict
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class ObjectTrackingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # 初始化跟踪相关变量
        self.model = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 跟踪状态变量
        self.prev_positions = defaultdict(lambda: None)
        self.speeds = defaultdict(float)
        self.current_vehicle_speed = 80.0  # 默认速度 km/h
        
        # 视频处理标志
        self.is_processing = False

    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('目标跟踪系统')
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # 模型加载部分
        model_group = QWidget()
        model_layout = QHBoxLayout()
        self.model_path_label = QLabel('未加载模型')
        self.load_model_btn = QPushButton('加载模型')
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)
        model_group.setLayout(model_layout)
        
        # 视频选择部分
        video_group = QWidget()
        video_layout = QHBoxLayout()
        self.video_path_label = QLabel('未选择视频')
        self.load_video_btn = QPushButton('选择视频')
        self.load_video_btn.clicked.connect(self.load_video)
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.load_video_btn)
        video_group.setLayout(video_layout)
        
        # 参数设置部分
        params_group = QWidget()
        params_layout = QVBoxLayout()
        
        # 速度设置
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel('基准速度(km/h):'))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(0, 200)
        self.speed_spinbox.setValue(80)
        self.speed_spinbox.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_spinbox)
        
        # 置信度阈值设置
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel('置信度阈值:'))
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setRange(1, 100)
        self.conf_spinbox.setValue(30)
        conf_layout.addWidget(self.conf_spinbox)
        
        params_layout.addLayout(speed_layout)
        params_layout.addLayout(conf_layout)
        params_group.setLayout(params_layout)
        
        # 控制按钮
        self.start_btn = QPushButton('开始')
        self.start_btn.clicked.connect(self.toggle_processing)
        self.start_btn.setEnabled(False)
        
        # 将所有组件添加到控制面板
        control_layout.addWidget(model_group)
        control_layout.addWidget(video_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.start_btn)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # 右侧视频显示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        
        # 将控制面板和视频显示添加到主布局
        layout.addWidget(control_panel)
        layout.addWidget(self.video_label)
        
        main_widget.setLayout(layout)

    def load_model(self):
        """加载YOLO模型"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PT文件 (*.pt)")
        if file_name:
            try:
                self.model = YOLO(file_name)
                self.model_path_label.setText(os.path.basename(file_name))
                self.check_start_conditions()
            except Exception as e:
                self.model_path_label.setText(f'加载失败: {str(e)}')

    def load_video(self):
        """加载视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                self.video_path_label.setText(os.path.basename(file_name))
                self.check_start_conditions()
            else:
                self.video_path_label.setText('视频加载失败')

    def update_speed(self, value):
        """更新基准速度"""
        self.current_vehicle_speed = float(value)

    def check_start_conditions(self):
        """检查是否满足开始处理的条件"""
        self.start_btn.setEnabled(self.model is not None and self.cap is not None)

    def toggle_processing(self):
        """切换处理状态"""
        if not self.is_processing:
            self.is_processing = True
            self.start_btn.setText('停止')
            self.timer.start(30)  # 约33fps
        else:
            self.is_processing = False
            self.start_btn.setText('开始')
            self.timer.stop()

    def update_frame(self):
        """更新视频帧"""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
            return

        # 获取视频尺寸
        frame_height, frame_width = frame.shape[:2]

        # 目标跟踪处理
        try:
            results = self.model.track(
                frame, 
                persist=True,
                conf=self.conf_spinbox.value() / 100.0,
                tracker="bytetrack.yaml"
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.cpu()

                # 计算每个目标的速度
                for box, track_id in zip(boxes, track_ids):
                    track_id = int(track_id)
                    current_pos = box[:2].numpy()
                    current_size = box[2:4].numpy()

                    if self.prev_positions[track_id] is not None:
                        # 计算像素距离
                        distance = np.linalg.norm(current_pos - self.prev_positions[track_id])
                        
                        # 透视校正系数计算
                        y_factor = 1.0 + (1.0 - current_pos[1] / frame_height)
                        relative_area = (current_size[0] * current_size[1]) / (frame_width * frame_height)
                        size_factor = 1.0 + (1.0 - min(relative_area * 1000, 0.9)) * 1.5
                        x_factor = 1.0 + 0.1 * abs(current_pos[0] / frame_width - 0.5)
                        
                        # 位置校正
                        position_factor = 1.0
                        if current_pos[0] > frame_width * 0.75 and current_pos[1] > frame_height * 0.75:
                            position_factor = 0.8

                        # 综合校正
                        correction_factor = y_factor * size_factor * x_factor * position_factor
                        distance_meters = distance * 0.1 * correction_factor
                        
                        # 速度计算
                        relative_speed = self.current_vehicle_speed * (1 - (distance_meters / 50))
                        if relative_speed < 0:
                            relative_speed = 0
                        
                        self.speeds[track_id] = self.speeds[track_id] * 0.7 + relative_speed * 0.3
                    
                    self.prev_positions[track_id] = current_pos

            # 绘制结果
            annotated_frame = results[0].plot()

            # 添加速度信息
            if results[0].boxes.id is not None:
                for box, track_id in zip(boxes, track_ids):
                    track_id = int(track_id)
                    if self.speeds[track_id] > 0:
                        x, y = int(box[0]), int(box[1])
                        speed_text = f"{self.speeds[track_id]:.1f} km/h"
                        cv2.putText(annotated_frame, speed_text, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 转换图像格式并显示
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

    def closeEvent(self, event):
        """关闭窗口时的清理工作"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ObjectTrackingUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()