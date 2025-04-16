# NeuroVision Qt

Универсальный проигрыватель видео с возможностью обработки различными нейросетевыми моделями в реальном времени через Qt-интерфейс.

## Структура проекта
NeuroVisionQt/
├── main.py # Основное приложение
├── plugin/ # Директория с моделями
│ ├── mediapipeModel # Реализация MediaPipe
│ └── yolo # Реализация YOLO
└── requirements.txt # Зависимости

## Требования

- Python 3.8+
- PyQt5
- OpenCV
- PyTorch (для YOLO)
- MediaPipe (для MediaPipe)

## В интерфейсе:

Нажмите "Выбрать видео" для загрузки файла

Активируйте нужные модели через чекбоксы

Используйте кнопки управления воспроизведением

Настройте скорость через слайдер

## Добавление новых моделей
Чтобы добавить новую модель:

Создайте новый Python-модуль в папке plugin/

Реализуйте обязательные методы:

```python
class MyModel:
    def process_frame(self, frame):
        """Обработка кадра"""

    
    def draw_skeleton(self, frame, points):
        """Отрисовка результатов"""
```


Напишите модель в plugin.py

```python
from NeuroVisionQt.my_model import MyModel

plugin = {
    "MyModel": MyModel(),
    # существующие модели...
}
```
