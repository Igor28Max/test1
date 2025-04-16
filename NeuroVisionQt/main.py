import sys
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QCheckBox, QMessageBox, QGroupBox, QButtonGroup, QSlider
)

from NeuroVisionQt.plugin import *


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)  # Отправка кадра для отображения
    finished_signal = pyqtSignal()  # Сигнал завершения видео
    position_changed = pyqtSignal(int, int)  # Текущая позиция и общая длина

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.cap = None
        self.active_models = {}
        self.loop_video = False
        self.is_paused = False
        self.speed = 1.0
        self.base_delay = 30
        self.last_frame_time = 0

        self.seeking = False
        self.target_frame = -1
        self.seek_lock = QMutex()

    def run(self):
        while self.is_running:
            self.seek_lock.lock()
            seeking = self.seeking
            target = self.target_frame
            self.seek_lock.unlock()

            if seeking and target >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                self.seek_lock.lock()
                self.seeking = False
                self.target_frame = -1
                self.is_paused = False
                self.seek_lock.unlock()
                continue

            if not self.is_paused:
                start_time = time.time() * 1000

                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.loop_video:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            self.finished_signal.emit()
                            break

                    for model_name, model in self.active_models.items():
                        points = model.process_frame(frame)
                        if points is not None:
                            model.draw_skeleton(frame, points)

                    self.change_pixmap_signal.emit(frame)
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.position_changed.emit(pos, total)

                if self.last_frame_time > 0:
                    elapsed = (time.time() * 1000) - self.last_frame_time
                    target_delay = self.base_delay / self.speed
                    remaining_delay = max(1, target_delay - elapsed)
                    self.msleep(int(remaining_delay))

                self.last_frame_time = time.time() * 1000
            else:
                self.msleep(50)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.quit()

    def update_active_models(self, models_dict):
        self.active_models = {name: model for name, model in models_dict.items()
                              if model is not None}

    def seek(self, position):
        """Безопасная установка позиции для перемотки"""
        self.seek_lock.lock()
        try:
            self.seeking = True
            self.target_frame = position
            self.is_paused = True  # Приостанавливаем обработку
        finally:
            self.seek_lock.unlock()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Pose")
        self.setGeometry(100, 100, 1000, 800)

        self.models = plugin
        self.active_models = {}

        self.video_thread = VideoThread()
        self.init_ui()

        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.finished_signal.connect(self.video_finished)
        self.video_thread.position_changed.connect(self.update_progress)

    def init_ui(self):

        # Основные виджеты
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)

        self.open_btn = QPushButton("📁 Выбрать видео", self)
        self.start_btn = QPushButton("▶ Старт", self)
        self.stop_btn = QPushButton("⏹ Стоп", self)
        self.loop_btn = QPushButton("🔁 Зациклить", self)
        self.loop_btn.setCheckable(True)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)  # 0.1x
        self.speed_slider.setMaximum(400)  # 2x
        self.speed_slider.setValue(100)  # 1x
        self.speed_label = QLabel("Скорость: 1.0x")

        self.model_group = QGroupBox("Активные модели")
        self.model_layout = QVBoxLayout()

        self.model_buttons = {}
        for model_name in self.models.keys():
            checkbox = QCheckBox(model_name, self)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.update_model_selection)
            self.model_buttons[model_name] = checkbox
            self.model_layout.addWidget(checkbox)

        self.model_group.setLayout(self.model_layout)

        self.status_label = QLabel("Статус: Остановлено", self)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.loop_btn)

        control_layout.addWidget(QLabel("Скорость:"))
        control_layout.addWidget(self.speed_slider)
        control_layout.addWidget(self.speed_label)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.time_label = QLabel("00:00 / 00:00")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.model_group)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.progress_slider)
        main_layout.addWidget(self.time_label)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.open_btn.clicked.connect(self.open_video)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.loop_btn.toggled.connect(self.toggle_loop)

        self.speed_slider.valueChanged.connect(self.change_speed)

        self.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-size: 14px;
                min-width: 100px;
            }
            QCheckBox {
                padding: 5px;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
        """)

        self.video_thread.position_changed.connect(self.update_progress)
        self.progress_slider.sliderMoved.connect(self.seek_video)

    def update_progress(self, pos, total):
        """Обновление прогресс-бара и времени"""
        if not self.video_thread.seeking and total > 0:
            self.progress_slider.blockSignals(True)  # Блокируем сигналы
            self.progress_slider.setMaximum(total)
            self.progress_slider.setValue(pos)
            self.progress_slider.blockSignals(False)

            fps = self.video_thread.cap.get(cv2.CAP_PROP_FPS) or 30
            current_time = self.frames_to_time(pos, fps)
            total_time = self.frames_to_time(total, fps)
            self.time_label.setText(f"{current_time} / {total_time}")

    def frames_to_time(self, frames, fps):

        seconds = int(frames / fps)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def seek_video(self, position):
        """Обработчик перемотки видео"""
        if hasattr(self, 'video_thread') and self.video_thread.cap and self.video_thread.cap.isOpened():
            self.video_thread.seek(position)

    def update_model_selection(self):
        self.active_models = {}
        for model_name, checkbox in self.model_buttons.items():
            if checkbox.isChecked():
                self.active_models[model_name] = self.models[model_name]

        self.video_thread.update_active_models(self.active_models)
        self.status_label.setText(f"Активные модели: {', '.join(self.active_models.keys())}")

    def update_image(self, cv_img):
        """Отображение обработанного кадра"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Конвертация OpenCV изображения в QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        ))

    def change_speed(self, value):
        speed = value / 100.0
        self.video_thread.speed = speed
        self.speed_label.setText(f"Скорость: {speed:.1f}x")

        if speed < 0.9:
            self.speed_label.setStyleSheet("color: blue;")
        elif speed > 1.1:
            self.speed_label.setStyleSheet("color: red;")
        else:
            self.speed_label.setStyleSheet("")

    def open_video(self):
        """Загрузка видеофайла"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Видео файлы (*.mp4 *.avi *.mov)"
        )
        if file_name:
            if self.video_thread.isRunning():
                self.video_thread.stop()

            self.video_thread.cap = cv2.VideoCapture(file_name)
            if self.video_thread.cap.isOpened():

                total_frames = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.progress_slider.setRange(0, total_frames)

                fps = self.video_thread.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.video_thread.base_delay = int(1000 / fps)
                else:
                    self.video_thread.base_delay = 30

                self.open_btn.setStyleSheet("background-color: #a0e0a0;")
                self.start_btn.setStyleSheet("")  # Сбрасываем зеленую подсветку
                self.status_label.setText(f"Видео загружено: {file_name.split('/')[-1]}")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось открыть видеофайл!")

    def start_processing(self):
        if not self.video_thread.cap or not self.video_thread.cap.isOpened():
            QMessageBox.warning(self, "Ошибка", "Сначала выберите видео!")
            return

        self.start_btn.setStyleSheet("")

        if not self.video_thread.isRunning():
            self.video_thread.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.video_thread.is_paused = False
        self.video_thread.is_running = True

        if not self.video_thread.isRunning():
            self.video_thread.start()

        self.status_label.setText(f"Обработка... | Модели: {', '.join(self.active_models.keys())}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_processing(self):
        if not self.video_thread.is_paused:  # Если видео играет, ставим на паузу
            self.video_thread.pause()
            self.status_label.setText(f"Пауза | Модели: {', '.join(self.active_models.keys())}")
            self.stop_btn.setText("▶ Продолжить")  # Меняем текст кнопки
        else:  # Если видео на паузе
            self.video_thread.resume()
            self.status_label.setText(f"Обработка... | Модели: {', '.join(self.active_models.keys())}")
            self.stop_btn.setText("⏸ Пауза")  # Возвращаем текст кнопки

    def toggle_loop(self, checked):
        self.video_thread.loop_video = checked
        self.loop_btn.setStyleSheet("background-color: #a0e0a0;" if checked else "")

    def video_finished(self):
        self.stop_processing()
        if not self.video_thread.loop_video:
            QMessageBox.information(self, "Информация", "Видео завершено!")

        self.start_btn.setEnabled(True)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
