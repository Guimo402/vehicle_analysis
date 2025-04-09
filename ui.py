import sys
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class VehicleSpeedTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车辆速度跟踪系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.model = None
        self.cap = None
        self.is_processing = False
        self.prev_positions = {}
        self.estimated_speeds = {}
        self.frame_time = 0.033  # 默认30fps
        self.camera_speed = 80.0  # 默认本车速度80km/h
        
        # 参数设置
        self.CONFIDENCE_THRESHOLD = 0.5
        self.DISTANCE_SCALE = 0.2
        self.SMOOTH_FACTOR = 0.5
        
        # 初始化UI
        self.initUI()
        
        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def initUI(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # 模型加载组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)
        self.load_model_btn = QPushButton("加载YOLO模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.model_path_label = QLabel("未加载模型")
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_path_label)
        
        # 视频加载组
        video_group = QGroupBox("视频设置")
        video_layout = QVBoxLayout(video_group)
        self.load_video_btn = QPushButton("加载视频")
        self.load_video_btn.clicked.connect(self.load_video)
        self.video_path_label = QLabel("未加载视频")
        video_layout.addWidget(self.load_video_btn)
        video_layout.addWidget(self.video_path_label)
        
        # 参数设置组
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        
        # 本车速度设置
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("本车速度(km/h):"))
        self.camera_speed_spin = QSpinBox()
        self.camera_speed_spin.setRange(0, 200)
        self.camera_speed_spin.setValue(80)
        self.camera_speed_spin.valueChanged.connect(self.update_camera_speed)
        speed_layout.addWidget(self.camera_speed_spin)
        params_layout.addLayout(speed_layout)
        
        # 置信度阈值设置
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.1)
        self.conf_spin.setValue(0.5)
        self.conf_spin.valueChanged.connect(self.update_confidence)
        conf_layout.addWidget(self.conf_spin)
        params_layout.addLayout(conf_layout)
        
        # 控制按钮
        self.start_btn = QPushButton("开始分析")
        self.start_btn.clicked.connect(self.toggle_processing)
        self.start_btn.setEnabled(False)
        
        # 预警信息显示
        alert_group = QGroupBox("预警信息")
        alert_layout = QVBoxLayout(alert_group)
        self.feedback_label = QTextEdit()
        self.feedback_label.setReadOnly(True)
        alert_layout.addWidget(self.feedback_label)
        
        # 添加所有组件到控制面板
        control_layout.addWidget(model_group)
        control_layout.addWidget(video_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(alert_group)
        control_layout.addStretch(1)
        
        # 视频显示区域
        self.video_label = QLabel("请先加载模型和视频文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #212121; color: #ffffff;")
        self.video_label.setMinimumSize(800, 600)
        
        # 添加控制面板和视频显示到主布局
        layout.addWidget(control_panel)
        layout.addWidget(self.video_label, 1)
        
        # 状态栏
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("系统准备就绪，请加载模型和视频。")
        
    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型文件", "", "PT文件 (*.pt)")
        if file_name:
            try:
                self.model = YOLO(file_name)
                self.model_path_label.setText(f"当前模型: {file_name}")
                self.statusBar.showMessage(f"模型加载成功！", 5000)
                self.check_start_conditions()
            except Exception as e:
                self.model_path_label.setText('模型加载失败')
                self.statusBar.showMessage(f"模型加载失败: {e}", 5000)
                self.model = None
                
    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_name:
            if self.cap is not None:
                self.cap.release()
            try:
                self.cap = cv2.VideoCapture(file_name)
                if not self.cap.isOpened():
                    raise IOError("无法打开视频文件")
                
                self.video_path_label.setText(f"当前视频: {file_name}")
                self.statusBar.showMessage(f"视频加载成功！", 5000)
                self.check_start_conditions()
                
                # 显示第一帧
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            except Exception as e:
                self.video_path_label.setText('视频加载失败')
                self.statusBar.showMessage(f"视频加载失败: {e}", 5000)
                self.cap = None
                
    def check_start_conditions(self):
        ready = self.model is not None and self.cap is not None and self.cap.isOpened()
        self.start_btn.setEnabled(ready)
        
    def toggle_processing(self):
        if not self.is_processing:
            if not (self.model and self.cap and self.cap.isOpened()):
                self.statusBar.showMessage("错误: 请确保已加载模型和视频", 5000)
                return
                
            self.prev_positions.clear()
            self.estimated_speeds.clear()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.is_processing = True
            self.start_btn.setText('停止分析')
            self.statusBar.showMessage("正在处理视频...")
            
            # 禁用参数调整
            for widget in self.findChildren((QSpinBox, QDoubleSpinBox)):
                widget.setEnabled(False)
            self.load_model_btn.setEnabled(False)
            self.load_video_btn.setEnabled(False)
            
            self.timer.start(int(self.frame_time * 1000))
        else:
            self.is_processing = False
            self.timer.stop()
            self.start_btn.setText('开始分析')
            self.statusBar.showMessage("处理已停止", 5000)
            
            # 启用参数调整
            for widget in self.findChildren((QSpinBox, QDoubleSpinBox)):
                widget.setEnabled(True)
            self.load_model_btn.setEnabled(True)
            self.load_video_btn.setEnabled(True)
            
    def update_camera_speed(self, value):
        self.camera_speed = value
        
    def update_confidence(self, value):
        self.CONFIDENCE_THRESHOLD = value
        
    def calculate_speed(self, prev_pos, current_pos, frame_time, frame_height, frame_width, current_size):
        # 计算像素距离
        distance = np.linalg.norm(current_pos - prev_pos)
        
        # 透视校正系数
        y_factor = 1.0 + (1.0 - current_pos[1] / frame_height) * 2.0
        relative_area = (current_size[0] * current_size[1]) / (frame_width * frame_height)
        size_factor = 1.0 + (1.0 - min(relative_area * 1000, 0.9)) * 2.0
        x_factor = 1.0 + 0.2 * abs(current_pos[0] / frame_width - 0.5)
        
        # 综合校正系数
        correction_factor = y_factor * size_factor * x_factor
        correction_factor = np.clip(correction_factor, 0.1, 20.0)
        
        # 计算实际距离
        distance_meters = distance * self.DISTANCE_SCALE * correction_factor
        
        # 计算速度
        speed_mps = distance_meters / frame_time
        speed_kmh = speed_mps * 3.6
        
        return speed_kmh
        
    def evaluate_danger(self, speed):
        # 评估危险等级
        if speed > self.camera_speed + 30:
            return "危险", (0, 0, 255)  # 红色
        elif speed > self.camera_speed + 10:
            return "注意", (0, 255, 255)  # 黄色
        else:
            return "安全", (50, 205, 50)  # 绿色
            
    def update_frame(self):
        if not self.is_processing or not self.cap or not self.cap.isOpened():
            if self.is_processing:
                self.toggle_processing()
            return
            
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_processing()
            self.statusBar.showMessage("视频播放完毕", 5000)
            return
            
        processed_frame = frame.copy()
        feedback_list = []
        
        try:
            # 目标检测与跟踪
            results = self.model.track(
                frame,
                persist=True,
                conf=self.CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],  # 车辆类别
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xywh = results[0].boxes.xywh.cpu()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                current_tracked_ids = set()
                
                for i, track_id in enumerate(track_ids):
                    current_tracked_ids.add(track_id)
                    box_xywh = boxes_xywh[i]
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    current_pos = box_xywh[:2].numpy()
                    current_size = box_xywh[2:4].numpy()
                    
                    speed_calculated = False
                    if track_id in self.prev_positions:
                        try:
                            speed = self.calculate_speed(
                                self.prev_positions[track_id],
                                current_pos,
                                self.frame_time,
                                frame.shape[0],
                                frame.shape[1],
                                current_size
                            )
                            
                            # 计算相对速度
                            relative_speed = speed - self.camera_speed
                            
                            # 平滑处理
                            if track_id in self.estimated_speeds:
                                old_speed = self.estimated_speeds[track_id]
                                self.estimated_speeds[track_id] = (
                                    old_speed * self.SMOOTH_FACTOR + 
                                    relative_speed * (1 - self.SMOOTH_FACTOR)
                                )
                            else:
                                self.estimated_speeds[track_id] = relative_speed
                                
                            speed_calculated = True
                            
                        except Exception as e:
                            self.estimated_speeds.pop(track_id, None)
                            
                    self.prev_positions[track_id] = current_pos
                    
                    # 评估危险等级
                    if speed_calculated and abs(self.estimated_speeds[track_id]) > 0.1:
                        risk_level, color = self.evaluate_danger(self.estimated_speeds[track_id])
                        label = f"ID:{track_id} {self.estimated_speeds[track_id]:.1f}km/h [{risk_level}]"
                    else:
                        risk_level, color = "未知", (255, 255, 255)
                        label = f"ID:{track_id} ...km/h"
                        
                    # 更新反馈信息
                    feedback_list.append(label)
                    
                    # 绘制边框和标签
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    label_y = y1 - 5 if y1 - 5 > label_height else y1 + label_height + 5
                    overlay = processed_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (x1, label_y - label_height - baseline),
                        (x1 + label_width, label_y),
                        (0, 0, 0),
                        -1
                    )
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
                    cv2.putText(
                        processed_frame,
                        label,
                        (x1, label_y - baseline + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                # 清除消失的目标
                disappeared_ids = set(self.estimated_speeds.keys()) - current_tracked_ids
                for gone_id in disappeared_ids:
                    self.estimated_speeds.pop(gone_id, None)
                    self.prev_positions.pop(gone_id, None)
                    
            self.display_frame(processed_frame)
            
            # 更新预警信息
            self.feedback_label.setPlainText("\n".join(feedback_list))
            
            # 检查危险情况
            if any("危险" in txt for txt in feedback_list):
                self.statusBar.showMessage("预警：检测到危险车辆！", 3000)
                
            # 显示处理速度
            end_time = time.time()
            processing_fps = 1.0 / (end_time - start_time)
            self.statusBar.showMessage(f"处理速度: {processing_fps:.1f} FPS", 1000)
            
        except Exception as e:
            print(f"处理帧时出错: {e}")
            self.display_frame(frame)
            
    def display_frame(self, frame):
        if frame is None:
            return
            
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"显示帧时出错: {e}")
            self.video_label.setText("无法显示帧")
            
    def closeEvent(self, event):
        self.is_processing = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = VehicleSpeedTracker()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()