import sys
import os
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QRadioButton, QLineEdit, QPushButton, QLabel, 
    QFileDialog, QSpinBox, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QFont, QIcon

# --- Constants ---
DEFAULT_WEIGHTS = "weights/best_tsm_topdown.pth"
DEFAULT_VARIANT = "large"
DEFAULT_SEGMENTS = 12

class WinFormApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity AI - Violence Detection System")
        self.setMinimumWidth(500)
        self.setup_ui()
        self.process = None

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_label = QLabel("HỆ THỐNG GIÁM SÁT BẠO LỰC")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #2c3e50;")
        main_layout.addWidget(header_label)

        # Info Box
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setStyleSheet("background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;")
        info_layout = QVBoxLayout(info_frame)
        model_info = QLabel(f"<b>Model:</b> TSM Large (86.43%)<br><b>Weights:</b> {DEFAULT_WEIGHTS}<br><b>Segments:</b> {DEFAULT_SEGMENTS}")
        model_info.setStyleSheet("font-size: 10pt; color: #34495e; border: none;")
        info_layout.addWidget(model_info)
        main_layout.addWidget(info_frame)

        # Group: Nguồn đầu vào
        source_group = QGroupBox("Cấu hình nguồn đầu vào")
        source_group.setStyleSheet("font-weight: bold;")
        source_layout = QVBoxLayout(source_group)

        # Radio 1: Camera
        cam_layout = QHBoxLayout()
        self.radio_cam = QRadioButton("Camera máy tính (ID)")
        self.radio_cam.setChecked(True)
        self.input_cam_id = QLineEdit("0")
        self.input_cam_id.setFixedWidth(50)
        cam_layout.addWidget(self.radio_cam)
        cam_layout.addWidget(self.input_cam_id)
        cam_layout.addStretch()
        source_layout.addLayout(cam_layout)

        # Radio 2: File
        file_layout = QHBoxLayout()
        self.radio_file = QRadioButton("Tệp video cục bộ")
        self.input_file_path = QLineEdit()
        self.input_file_path.setPlaceholderText("Đường dẫn đến file...")
        self.btn_browse = QPushButton("Duyệt file")
        self.btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.radio_file)
        file_layout.addWidget(self.input_file_path)
        file_layout.addWidget(self.btn_browse)
        source_layout.addLayout(file_layout)

        # Radio 3: DroidCam
        droid_layout = QHBoxLayout()
        self.radio_droid = QRadioButton("DroidCam (IP Stream)")
        self.input_ip = QLineEdit("192.168.1.10")
        self.input_ip.setPlaceholderText("IP Address")
        self.input_port = QLineEdit("4747")
        self.input_port.setFixedWidth(60)
        self.input_port.setPlaceholderText("Port")
        droid_layout.addWidget(self.radio_droid)
        droid_layout.addWidget(self.input_ip)
        droid_layout.addWidget(QLabel(":"))
        droid_layout.addWidget(self.input_port)
        source_layout.addLayout(droid_layout)

        main_layout.addWidget(source_group)

        # Settings Group
        settings_group = QGroupBox("Cài đặt hiệu năng")
        settings_layout = QHBoxLayout(settings_group)
        settings_layout.addWidget(QLabel("Độ phân giải (imgsz):"))
        self.input_imgsz = QSpinBox()
        self.input_imgsz.setRange(320, 640)
        self.input_imgsz.setSingleStep(32)
        self.input_imgsz.setValue(416)
        settings_layout.addWidget(self.input_imgsz)
        settings_layout.addStretch()
        main_layout.addWidget(settings_group)

        # Action Button
        self.btn_start = QPushButton("BẮT ĐẦU GIÁM SÁT")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.btn_start.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.btn_start)

        # Status Label
        self.status_label = QLabel("Sẵn sàng.")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        main_layout.addWidget(self.status_label)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.input_file_path.setText(file_path)
            self.radio_file.setChecked(True)

    def start_pipeline(self):
        # Build command
        cmd = [
            sys.executable, "src/buoc6_main_pipeline.py",
            "--tsm-weights", DEFAULT_WEIGHTS,
            "--tsm-variant", DEFAULT_VARIANT,
            "--num-segments", str(DEFAULT_SEGMENTS),
            "--imgsz", str(self.input_imgsz.value()),
            "--detector-mode", "track",
            "--tsm-only-on-threshold", "0.70",
            "--decision-mode", "tsm_only",
            "--show"
        ]

        if self.radio_cam.isChecked():
            cmd.extend(["--camera-id", self.input_cam_id.text()])
        elif self.radio_file.isChecked():
            path = self.input_file_path.text()
            if not path:
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file video!")
                return
            cmd.extend(["--video-path", path])
        elif self.radio_droid.isChecked():
            ip = self.input_ip.text()
            port = self.input_port.text()
            url = f"http://{ip}:{port}/video"
            cmd.extend(["--video-path", url])

        # Execute
        self.status_label.setText("Đang khởi chạy pipeline...")
        self.btn_start.setEnabled(False)
        
        # We use QProcess to keep UI responsive
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self.process.finished.connect(self.on_finished)
        self.process.start(cmd[0], cmd[1:])
        
        if not self.process.waitForStarted(3000):
            QMessageBox.critical(self, "Lỗi", "Không thể khởi động chương trình!")
            self.on_finished()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.status_label.setText("Đã dừng.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WinFormApp()
    window.show()
    sys.exit(app.exec())
