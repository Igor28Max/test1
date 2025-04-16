import cv2
import torch # version 1.13
import numpy as np
import mediapipe as mp # 0.10.9
import torchvision# 0.14

from utils.datasets import letterbox


class MediaPipePose:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


    def process_predictions(self, pred, conf_thres=0.25, iou_thres=0.45):
        """
        Обрабатывает выход модели:
          - Убирает предсказания с низкой уверенность (индекс 4).
          - Применяет Non-Maximum Suppression (NMS) для bbox.
        """
        # Убираем размер батча
        pred = pred[0]  # теперь форма (15300, 57)

        # Фильтрация по порогу уверенности объекта (индекс 4)
        mask = pred[:, 4] > conf_thres
        pred = pred[mask]

        if pred.shape[0] == 0:
            return []

        # Извлекаем ограничивающие прямоугольники
        boxes = pred[:, :4]
        # Считаем итоговую уверенность: объектная уверенность * уверенность класса
        scores = pred[:, 4] * pred[:, 5]

        # Применяем NMS (для этого могут потребоваться координаты в формате [x1, y1, x2, y2])
        # Если у вас bbox заданы как (x, y, w, h), преобразуем:
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h

        keep = torchvision.ops.nms(boxes_xyxy, scores, iou_thres)
        pred = pred[keep]

        return pred

    def scale_coords(self, coords, ratio, pad, original_shape):
        """
        Преобразует координаты, полученные на letterbox-изображении, в координаты исходного изображения.
        :param coords: numpy-массив координат (N, 2) или (N, 3) (x, y, [conf])
        :param ratio: коэффициент масштабирования, возвращаемый letterbox
        :param pad: tuple (pad_x, pad_y), отступы, возвращаемые letterbox
        :param original_shape: кортеж (h, w) исходного изображения
        :return: преобразованные координаты
        """
        coords = np.atleast_2d(coords)  # если coords уже (N,2), то ничего не меняется

        # Распаковываем отступы
        pad_x, pad_y = pad  # убедитесь, что pad передается как кортеж (pad_x, pad_y)

        # Преобразуем координаты обратно к размерам исходного изображения
        coords[:, 0] = (coords[:, 0] - pad_x) / ratio[0]
        coords[:, 1] = (coords[:, 1] - pad_y) / ratio[1]
        return coords

    def process_frame(self, frame):
        """
        Обрабатывает кадр с помощью Mediapipe Pose:
        - преобразует кадр в RGB,
        - выполняет инференс,
        - рисует найденные ключевые точки (синим цветом) на кадре.
        Возвращает список точек в формате [(x, y, visibility), ...].
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Shape:", img_rgb.shape, "Dtype:", img_rgb.dtype)
        results = self.pose.process(img_rgb)
        keypoints = []
        h, w, _ = frame.shape

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x, y, visibility = lm.x * w, lm.y * h, lm.visibility
                keypoints.append((x, y, visibility))
                cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
        else:
            # Если не обнаружено, заполняем список нулями (количество точек в Mediapipe = 33)
            keypoints = [(0, 0, 0)] * 33

        return keypoints

    def draw_skeleton(self, frame, landmarks, flag=False):
        """
        Отрисовывает линии скелета для ключевых точек Mediapipe.
        Используем набор соединений, предоставленный Mediapipe.
        """
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1, *_ = landmarks[start_idx]
                x2, y2, *_ = landmarks[end_idx]
                if flag:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                else:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

