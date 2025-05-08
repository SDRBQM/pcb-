import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                            QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, 
                            QSlider, QMessageBox, QStackedWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('用户登录')
        
        layout = QVBoxLayout()
        
        self.username_label = QLabel('用户名:')
        self.username_input = QLineEdit()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        
        self.password_label = QLabel('密码:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        
        self.login_button = QPushButton('登录')
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)
        
        self.register_button = QPushButton('注册新账号')
        self.register_button.clicked.connect(self.show_register)
        layout.addWidget(self.register_button)
        
        self.setLayout(layout)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, '错误', '用户名和密码不能为空!')
            return
            
        if os.path.exists('users.txt'):
            with open('users.txt', 'r') as f:
                for line in f:
                    stored_user, stored_pass = line.strip().split(',')
                    if username == stored_user and password == stored_pass:
                        QMessageBox.information(self, '成功', '登录成功!')
                        self.parent.username = username
                        self.parent.stacked_widget.setCurrentIndex(1)
                        return
                        
        QMessageBox.warning(self, '错误', '用户名或密码错误!')
    
    def show_register(self):
        self.parent.stacked_widget.setCurrentIndex(2)

class RegisterWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('用户注册')
        
        layout = QVBoxLayout()
        
        self.username_label = QLabel('用户名:')
        self.username_input = QLineEdit()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        
        self.password_label = QLabel('密码:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        
        self.confirm_label = QLabel('确认密码:')
        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.confirm_label)
        layout.addWidget(self.confirm_input)
        
        self.register_button = QPushButton('注册')
        self.register_button.clicked.connect(self.register)
        layout.addWidget(self.register_button)
        
        self.back_button = QPushButton('返回登录')
        self.back_button.clicked.connect(self.back_to_login)
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)
    
    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, '错误', '用户名和密码不能为空!')
            return
            
        if password != confirm:
            QMessageBox.warning(self, '错误', '两次输入的密码不一致!')
            return
            
        if os.path.exists('users.txt'):
            with open('users.txt', 'r') as f:
                for line in f:
                    stored_user, _ = line.strip().split(',')
                    if username == stored_user:
                        QMessageBox.warning(self, '错误', '用户名已存在!')
                        return
                        
        with open('users.txt', 'a') as f:
            f.write(f'{username},{password}\n')
            
        QMessageBox.information(self, '成功', '注册成功!')
        self.back_to_login()
    
    def back_to_login(self):
        self.username_input.clear()
        self.password_input.clear()
        self.confirm_input.clear()
        self.parent.stacked_widget.setCurrentIndex(0)

class DetectionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.model = None
        self.model_path = None
        self.image_path = None
        self.confidence = 0.5
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('缺陷检测系统')
        
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # 左侧面板 - 控制区
        self.model_label = QLabel('选择模型:')
        self.model_path_label = QLabel('未选择模型')
        self.model_path_label.setWordWrap(True)
        self.select_model_button = QPushButton('浏览模型文件')
        self.select_model_button.clicked.connect(self.select_model)
        
        self.load_model_button = QPushButton('加载模型')
        self.load_model_button.clicked.connect(self.load_model)
        
        self.image_label = QLabel('选择图片:')
        self.select_image_button = QPushButton('浏览图片')
        self.select_image_button.clicked.connect(self.select_image)
        
        self.confidence_label = QLabel(f'置信度阈值: {self.confidence}')
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(self.confidence * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        
        self.detect_button = QPushButton('开始检测')
        self.detect_button.clicked.connect(self.detect)
        self.detect_button.setEnabled(False)
        
        self.save_button = QPushButton('保存结果')
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_path_label)
        left_layout.addWidget(self.select_model_button)
        left_layout.addWidget(self.load_model_button)
        left_layout.addStretch(1)
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.select_image_button)
        left_layout.addStretch(1)
        left_layout.addWidget(self.confidence_label)
        left_layout.addWidget(self.confidence_slider)
        left_layout.addStretch(1)
        left_layout.addWidget(self.detect_button)
        left_layout.addWidget(self.save_button)
        left_layout.addStretch(3)
        
        # 右侧面板 - 图像显示区
        self.original_image = QLabel('原始图像将显示在这里')
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setStyleSheet("border: 1px solid black;")
        
        self.result_image = QLabel('检测结果将显示在这里')
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setStyleSheet("border: 1px solid black;")
        
        right_layout.addWidget(QLabel('原始图像:'))
        right_layout.addWidget(self.original_image)
        right_layout.addWidget(QLabel('检测结果:'))
        right_layout.addWidget(self.result_image)
        
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)
        
        self.setLayout(main_layout)
    
    def select_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", 
            "",  # 初始目录
            "PyTorch Model Files (*.pt)", 
            options=options
        )
        
        if file_path:
            self.model_path = file_path
            # 显示模型文件名和部分路径
            dirname, basename = os.path.split(file_path)
            display_text = f"{basename}\n(位于: {dirname[:20]}...)" if len(dirname) > 20 else f"{basename}\n(位于: {dirname})"
            self.model_path_label.setText(display_text)
    
    def load_model(self):
        if not self.model_path:
            QMessageBox.warning(self, '警告', '请先选择模型文件!')
            return
            
        try:
            self.model = YOLO(self.model_path)
            QMessageBox.information(self, '成功', '模型加载成功!')
            if self.image_path:
                self.detect_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载模型失败: {str(e)}')
            self.model = None
    
    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp)", 
            options=options
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(self.original_image, file_path)
            if self.model:
                self.detect_button.setEnabled(True)
    
    def update_confidence(self, value):
        self.confidence = value / 100
        self.confidence_label.setText(f'置信度阈值: {self.confidence:.2f}')
    
    def detect(self):
        if not self.model or not self.image_path:
            QMessageBox.warning(self, '警告', '请先加载模型并选择图片!')
            return
            
        try:
            results = self.model(self.image_path, conf=self.confidence)
            
            for result in results:
                img_with_boxes = result.plot()
                
                height, width, channel = img_with_boxes.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_with_boxes.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_img)
                
                self.result_image.setPixmap(pixmap.scaled(
                    self.result_image.width(), 
                    self.result_image.height(), 
                    Qt.KeepAspectRatio
                ))
                
                self.result_data = img_with_boxes
                self.save_button.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, '错误', f'检测过程中出错: {str(e)}')
    
    def save_result(self):
        if not hasattr(self, 'result_data'):
            QMessageBox.warning(self, '警告', '没有可保存的结果!')
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", 
            "JPEG Files (*.jpg);;PNG Files (*.png)", 
            options=options
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_data)
                QMessageBox.information(self, '成功', f'结果已保存到: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'保存失败: {str(e)}')
    
    def display_image(self, label, image_path):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(
            label.width(), 
            label.height(), 
            Qt.KeepAspectRatio
        ))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.username = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('YOLOv8 缺陷检测系统')
        self.setGeometry(100, 100, 1000, 600)
        
        self.stacked_widget = QStackedWidget()
        
        self.login_window = LoginWindow(self)
        self.detection_window = DetectionWindow(self)
        self.register_window = RegisterWindow(self)
        
        self.stacked_widget.addWidget(self.login_window)
        self.stacked_widget.addWidget(self.detection_window)
        self.stacked_widget.addWidget(self.register_window)
        
        self.setCentralWidget(self.stacked_widget)
        
        self.stacked_widget.setCurrentIndex(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())