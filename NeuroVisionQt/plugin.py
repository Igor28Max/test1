from NeuroVisionQt.mediapipeModel.mediapipeModel import *
from NeuroVisionQt.yolo.yolo_7 import *


plugin = {
    "YOLOv7": Yolo7(),
    "YOLOv7_2": Yolo7(),
    "YOLOv7_3": Yolo7(),
    "YOLOv7_4": Yolo7(),
    "YOLOv7_5": Yolo7(),
    "MediaPipe": MediaPipePose()
}

