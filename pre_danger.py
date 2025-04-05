# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                             QSpinBox, QGroupBox, QDoubleSpinBox, QStatusBar,
                             QSizePolicy, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QBrush
from ultralytics import YOLO
from collections import defaultdict
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ObjectTrackingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 初始化速度估计参数 ---
        self.CAMERA_SPEED_KMH = 80.0
        self.SMOOTH_FACTOR = 0.5
        self.DISTANCE_SCALE = 0.2
        self.RELATIVE_SPEED_DENOMINATOR = 30.0
        self.CONFIDENCE_THRESHOLD = 0.3

        # --- 内部状态变量 ---
        self.FRAME_H = None
        self.FRAME_W = None
        self.DELTA_T = 1.0 / 30.0

        # --- 跟踪状态变量 ---
        self.prev_positions = defaultdict(lambda: None)
        self.estimated_speeds = defaultdict(float)

        # --- 其他变量 ---
        self.model = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_processing = False

        self.initUI()

    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('智能车辆分析与预警系统')
        self.setGeometry(100, 100, 1450, 950)

        # --- 设置主窗口背景 ---
        palette = self.palette()
        gradient = "qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #E1F5FE, stop:1 #B3E5FC)"
        palette.setBrush(QPalette.Window, QBrush(QColor(240, 240, 240)))  # Light gray background
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # --- 左侧控制面板 ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(450)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(20)
        control_layout.setAlignment(Qt.AlignTop)
        control_panel.setStyleSheet("background-color: #f9f9f9; border-radius: 10px; padding: 15px;")

        # --- 模型加载组 ---
        model_group = QGroupBox("模型设置")
        self.style_group_box(model_group)
        model_layout = QVBoxLayout(model_group)
        self.model_path_label = QLabel('请加载 YOLO 模型 (.pt)')
        self.model_path_label.setWordWrap(True)
        self.style_label(self.model_path_label)
        self.load_model_btn = QPushButton('选择模型文件')
        self.style_button(self.load_model_btn)
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)

        # --- 视频选择组 ---
        video_group = QGroupBox("视频设置")
        self.style_group_box(video_group)
        video_layout = QVBoxLayout(video_group)
        self.video_path_label = QLabel('请选择视频文件')
        self.video_path_label.setWordWrap(True)
        self.style_label(self.video_path_label)
        self.load_video_btn = QPushButton('选择视频文件')
        self.style_button(self.load_video_btn)
        self.load_video_btn.clicked.connect(self.load_video)
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.load_video_btn)

        # --- 参数设置组 ---
        params_group = QGroupBox("速度估计参数调整")
        self.style_group_box(params_group)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(15)

        # 基准速度
        speed_layout = QHBoxLayout()
        speed_label = QLabel('基准速度 (km/h):')
        self.style_label(speed_label)
        speed_layout.addWidget(speed_label)
        self.speed_spinbox = QSpinBox()
        self.style_spinbox(self.speed_spinbox)
        self.speed_spinbox.setRange(0, 250)
        self.speed_spinbox.setValue(int(self.CAMERA_SPEED_KMH))
        self.speed_spinbox.valueChanged.connect(lambda v: setattr(self, 'CAMERA_SPEED_KMH', float(v)))
        speed_layout.addWidget(self.speed_spinbox)
        params_layout.addLayout(speed_layout)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel('置信度阈值:')
        self.style_label(conf_label)
        conf_layout.addWidget(conf_label)
        self.conf_spinbox = QDoubleSpinBox()
        self.style_spinbox(self.conf_spinbox)
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(self.CONFIDENCE_THRESHOLD)
        self.conf_spinbox.valueChanged.connect(lambda v: setattr(self, 'CONFIDENCE_THRESHOLD', v))
        conf_layout.addWidget(self.conf_spinbox)
        params_layout.addLayout(conf_layout)

        # 平滑因子
        smooth_layout = QHBoxLayout()
        smooth_label = QLabel('平滑因子 (越小越平滑):')
        self.style_label(smooth_label)
        smooth_layout.addWidget(smooth_label)
        self.smooth_spinbox = QDoubleSpinBox()
        self.style_spinbox(self.smooth_spinbox)
        self.smooth_spinbox.setRange(0.01, 0.99)
        self.smooth_spinbox.setSingleStep(0.05)
        self.smooth_spinbox.setValue(self.SMOOTH_FACTOR)
        self.smooth_spinbox.valueChanged.connect(lambda v: setattr(self, 'SMOOTH_FACTOR', v))
        smooth_layout.addWidget(self.smooth_spinbox)
        params_layout.addLayout(smooth_layout)

        # 距离缩放因子
        dist_scale_layout = QHBoxLayout()
        dist_scale_label = QLabel('距离缩放因子:')
        self.style_label(dist_scale_label)
        dist_scale_layout.addWidget(dist_scale_label)
        self.dist_scale_spinbox = QDoubleSpinBox()
        self.style_spinbox(self.dist_scale_spinbox)
        self.dist_scale_spinbox.setRange(0.01, 2.0)
        self.dist_scale_spinbox.setSingleStep(0.01)
        self.dist_scale_spinbox.setDecimals(3)
        self.dist_scale_spinbox.setValue(self.DISTANCE_SCALE)
        self.dist_scale_spinbox.valueChanged.connect(lambda v: setattr(self, 'DISTANCE_SCALE', v))
        dist_scale_layout.addWidget(self.dist_scale_spinbox)
        params_layout.addLayout(dist_scale_layout)

        # 相对速度分母
        rel_denom_layout = QHBoxLayout()
        rel_denom_label = QLabel('相对速度分母:')
        self.style_label(rel_denom_label)
        rel_denom_layout.addWidget(rel_denom_label)
        self.rel_denom_spinbox = QDoubleSpinBox()
        self.style_spinbox(self.rel_denom_spinbox)
        self.rel_denom_spinbox.setRange(1.0, 200.0)
        self.rel_denom_spinbox.setSingleStep(1.0)
        self.rel_denom_spinbox.setValue(self.RELATIVE_SPEED_DENOMINATOR)
        self.rel_denom_spinbox.valueChanged.connect(lambda v: setattr(self, 'RELATIVE_SPEED_DENOMINATOR', v))
        rel_denom_layout.addWidget(self.rel_denom_spinbox)
        params_layout.addLayout(rel_denom_layout)

        # --- 控制按钮 ---
        self.start_btn = QPushButton('开始分析')
        self.style_button(self.start_btn, bold=True, color="#4CAF50")  # Green start button
        self.start_btn.clicked.connect(self.toggle_processing)
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(50)

        # --- 预警信息显示组 ---
        alert_group = QGroupBox("预警信息")
        self.style_group_box(alert_group)
        alert_layout = QVBoxLayout(alert_group)
        self.feedback_label = QTextEdit()
        self.feedback_label.setReadOnly(True)
        self.feedback_label.setStyleSheet("font-size: 14px;")
        alert_layout.addWidget(self.feedback_label)

        # --- 将所有组件添加到控制面板布局 ---
        control_layout.addWidget(model_group)
        control_layout.addWidget(video_group)
        control_layout.addWidget(params_group)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(alert_group)
        control_layout.addStretch(1)

        # --- 右侧视频显示 ---
        self.video_label = QLabel("请先加载模型和视频文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #212121; color: #ffffff; border-radius: 10px; padding: 10px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(800, 600)

        # --- 将控制面板和视频显示添加到主布局 ---
        layout.addWidget(control_panel)
        layout.addWidget(self.video_label, 1)

        # --- 状态栏 ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setStyleSheet("background-color: #e0e0e0; color: #333;")
        self.statusBar.showMessage("系统准备就绪，请加载模型和视频。", 5000)

    def style_group_box(self, group_box):
        group_box.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                border: 2px solid #42A5F5;
                border-radius: 7px;
                margin-top: 15px;
                background-color: #e3f2fd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #1976D2;
            }
        """)

    def style_label(self, label):
        label.setStyleSheet("font-size: 14px; color: #333;")

    def style_button(self, button, bold=False, color="#1976D2"):
        font_weight = "bold" if bold else "normal"
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: {font_weight};
            }}
            QPushButton:hover {{
                background-color: #1565C0;
            }}
            QPushButton:disabled {{
                background-color: #90CAF9;
                color: #e0e0e0;
                border: none;
            }}
        """)

    def style_spinbox(self, spinbox):
        spinbox.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #757575;
                border-radius: 3px;
                padding: 5px;
                font-size: 14px;
            }
        """)

    def load_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型文件", "", "PT文件 (*.pt)")
        if file_name:
            try:
                self.statusBar.showMessage(f"正在加载模型: {os.path.basename(file_name)}...")
                QApplication.processEvents()
                self.model = YOLO(file_name)
                _ = self.model(np.zeros((64, 64, 3), dtype=np.uint8))
                self.model_path_label.setText(f"当前模型: {os.path.basename(file_name)}")
                self.statusBar.showMessage(f"模型 '{os.path.basename(file_name)}' 加载成功！", 5000)
                self.check_start_conditions()
            except Exception as e:
                self.model_path_label.setText('模型加载失败，请重试')
                self.statusBar.showMessage(f"模型加载失败: {e}", 0)
                self.model = None
                QMessageBox.critical(self, "模型加载错误", f"加载模型时发生错误:\n{e}")

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_name:
            if self.is_processing:
                self.toggle_processing()

            if self.cap is not None:
                self.cap.release()

            try:
                self.cap = cv2.VideoCapture(file_name)
                if not self.cap.isOpened():
                    raise IOError("无法打开视频文件")

                self.FRAME_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.FRAME_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    self.DELTA_T = 1.0 / 30.0
                    self.statusBar.showMessage("警告: 无法获取视频FPS，将假设为30", 3000)
                else:
                    self.DELTA_T = 1.0 / fps

                self.video_path_label.setText(f"当前视频: {os.path.basename(file_name)}")
                self.statusBar.showMessage(f"视频 '{os.path.basename(file_name)}' 加载成功 (FPS: {fps:.2f})", 5000)
                self.check_start_conditions()

                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                else:
                    self.video_label.setText("无法读取视频帧")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            except Exception as e:
                self.video_path_label.setText('视频加载失败，请重试')
                self.statusBar.showMessage(f"视频加载失败: {e}", 0)
                self.cap = None
                QMessageBox.critical(self, "视频加载错误", f"加载视频时发生错误:\n{e}")

    def check_start_conditions(self):
        ready = self.model is not None and self.cap is not None and self.cap.isOpened()
        self.start_btn.setEnabled(ready)
        if not ready and self.is_processing:
            self.toggle_processing()
        elif not ready:
            self.start_btn.setText('开始分析')

    def toggle_processing(self):
        if not self.is_processing:
            if not (self.model and self.cap and self.cap.isOpened()):
                self.statusBar.showMessage("错误: 请确保已成功加载模型和视频", 5000)
                QMessageBox.warning(self, "无法开始", "请先成功加载YOLO模型和视频文件。")
                return

            self.prev_positions.clear()
            self.estimated_speeds.clear()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.is_processing = True
            self.start_btn.setText('停止分析')
            self.style_button(self.start_btn, bold=True, color="#f44336")  # Red stop button
            self.statusBar.showMessage("正在处理视频...")
            for widget in self.findChildren((QSpinBox, QDoubleSpinBox)):
                widget.setEnabled(False)
            self.load_model_btn.setEnabled(False)
            self.load_video_btn.setEnabled(False)

            timer_interval = int(self.DELTA_T * 1000) if self.DELTA_T > 0 else 33
            self.timer.start(max(1, timer_interval))
        else:
            self.is_processing = False
            self.timer.stop()
            self.start_btn.setText('开始分析')
            self.style_button(self.start_btn, bold=True, color="#4CAF50")
            self.statusBar.showMessage("处理已停止", 5000)
            for widget in self.findChildren((QSpinBox, QDoubleSpinBox)):
                widget.setEnabled(True)
            self.load_model_btn.setEnabled(True)
            self.load_video_btn.setEnabled(True)

    def evaluate_danger(self, speed, relative_area):
        """
        根据车辆速度和目标在图像中的相对面积（作为距离近似指标）来评估危险等级。
        danger_score = estimated_speed * relative_area
        阈值可根据实际情况进行调试：
        - danger_score > 10 认为危险（红色）
        - danger_score > 5  认为注意（黄色）
        - 否则为安全（绿色）
        """
        danger_score = speed * relative_area
        if danger_score > 10:
            return "危险", (0, 0, 255)
        elif danger_score > 5:
            return "注意", (0, 255, 255)
        else:
            return "安全", (50, 205, 50)

    def update_frame(self):
        if not self.is_processing or not self.cap or not self.cap.isOpened():
            if self.is_processing:
                self.toggle_processing()
            return

        start_frame_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            self.toggle_processing()
            self.statusBar.showMessage("视频播放完毕，处理已停止。", 5000)
            return

        processed_frame = frame.copy()
        feedback_list = []

        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.CONFIDENCE_THRESHOLD,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
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
                    if self.prev_positions[track_id] is not None:
                        try:
                            distance = np.linalg.norm(current_pos - self.prev_positions[track_id])
                            if self.FRAME_H is None or self.FRAME_W is None:
                                continue

                            y_factor = 1.0 + (1.0 - current_pos[1] / self.FRAME_H) * 2.0
                            relative_area = (current_size[0] * current_size[1]) / (self.FRAME_W * self.FRAME_H)
                            size_factor = 1.0 + (1.0 - min(relative_area * 1000, 0.9)) * 2.0
                            x_factor = 1.0 + 0.2 * abs(current_pos[0] / self.FRAME_W - 0.5)
                            position_factor = 1.0
                            if current_pos[0] > self.FRAME_W * 0.75 and current_pos[1] > self.FRAME_H * 0.75:
                                position_factor = 0.7

                            correction_factor = y_factor * size_factor * x_factor * position_factor
                            correction_factor = np.clip(correction_factor, 0.1, 20.0)

                            distance_meters = distance * self.DISTANCE_SCALE * correction_factor

                            # 原先的速度计算：仅依赖于位移（速度与目标相对运动）
                            relative_speed = self.CAMERA_SPEED_KMH * (1 - (distance_meters / self.RELATIVE_SPEED_DENOMINATOR))
                            relative_speed = max(0, relative_speed)

                            old_speed = self.estimated_speeds.get(track_id, relative_speed)
                            self.estimated_speeds[track_id] = old_speed * self.SMOOTH_FACTOR + relative_speed * (1 - self.SMOOTH_FACTOR)
                            self.estimated_speeds[track_id] = max(0, min(200, self.estimated_speeds[track_id]))
                            speed_calculated = True

                        except Exception as e:
                            self.estimated_speeds.pop(track_id, None)

                    self.prev_positions[track_id] = current_pos

                    # 如果速度已经计算，同时确保相对面积不为0
                    if speed_calculated and self.estimated_speeds[track_id] > 0.1:
                        # 使用当前目标的相对面积来辅助判断危险等级
                        current_area = (current_size[0] * current_size[1]) / (self.FRAME_W * self.FRAME_H)
                        risk_level, color = self.evaluate_danger(self.estimated_speeds[track_id], current_area)
                        label = f"ID:{track_id} {self.estimated_speeds[track_id]:.1f}km/h [{risk_level}]"
                    else:
                        risk_level, color = "未知", (255, 255, 255)
                        label = f"ID:{track_id} ...km/h"

                    feedback_list.append(label)

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 - 5 if y1 - 5 > label_height else y1 + label_height + 5
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (x1, label_y - label_height - baseline), (x1 + label_width, label_y), (0, 0, 0), -1)
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
                    cv2.putText(processed_frame, label, (x1, label_y - baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                disappeared_ids = set(self.estimated_speeds.keys()) - current_tracked_ids
                for gone_id in disappeared_ids:
                    self.estimated_speeds.pop(gone_id, None)
                    self.prev_positions.pop(gone_id, None)

            self.display_frame(processed_frame)
            self.feedback_label.setPlainText("\n".join(feedback_list))
            if any("危险" in txt for txt in feedback_list):
                self.statusBar.showMessage("预警：检测到危险车辆！", 3000)
            else:
                self.statusBar.showMessage("正在处理...", 1000)

            end_frame_time = time.time()
            processing_fps = 1.0 / (end_frame_time - start_frame_time) if (end_frame_time - start_frame_time) > 0 else 0
            self.statusBar.showMessage(f"正在处理... FPS: {processing_fps:.1f}", 1000)

        except Exception as e:
            print(f"处理帧时发生严重错误: {e}")
            self.display_frame(frame)
            self.toggle_processing()
            QMessageBox.critical(self, "处理错误", f"处理视频帧时发生错误:\n{e}")


    def display_frame(self, frame):
        if frame is None:
            return
        try:
            if self.FRAME_H is None or self.FRAME_W is None:
                self.FRAME_H, self.FRAME_W = frame.shape[:2]

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            if h == 0 or w == 0:
                return
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"显示帧时出错: {e}")
            self.video_label.setText("无法显示帧")

    def closeEvent(self, event):
        self.is_processing = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        print("应用程序关闭，资源已释放。")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ObjectTrackingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
