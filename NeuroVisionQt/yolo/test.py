class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_running = True  # Флаг работы потока
        self._is_paused = False  # Флаг паузы
        self.cap = None
        self.active_models = {}
        self.loop_video = False

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, value):
        self._is_running = value

    @property
    def is_paused(self):
        return self._is_paused

    @is_paused.setter
    def is_paused(self, value):
        self._is_paused = value

    def run(self):
        while self.is_running:
            if self.is_paused:
                self.msleep(100)
                continue

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
                    if model_name == "YOLOv7":
                        yolo_points = model.process_frame_yolo(frame)
                        if yolo_points is not None:
                            model.draw_yolo_skeleton(frame, yolo_points)
                    elif model_name == "MediaPipe":
                        frame = model.process_frame(frame)

                self.change_pixmap_signal.emit(frame)
            else:
                self.msleep(50)

    def pause(self):
        """Поставить обработку на паузу"""
        self.is_paused = True

    def resume(self):
        """Продолжить обработку"""
        self.is_paused = False

    def stop(self):
        """Полная остановка потока"""
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
        self.quit()
