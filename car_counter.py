import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# для видео
cap = cv2.VideoCapture('./CarsVideo/Cars1.mp4')

model = YOLO('./YOLO_Weights/yolov8n.pt')

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "parking meter",
               "og", "horse", "sheep", "cow", "elephant", "elephant", "bear", "zebra", "giraffe", "backpack",
               "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "frisbee", "skis", "snowboard",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich",
               "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
               "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave"
                                                                                                            "oven",
               "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
               "toothbrush"]

mask = cv2.imread('mask.png')

# отлеживание
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ограничеие - линия
limits = [500, 461, 1280, 461 ]
total_count = []

while True:
    # Запуск камеры
    sucsess, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)
    results = model(img_region, stream=True)

    detections = np.empty((0, 5))
    # Нахожждение границ
    for res in results:
        boxes = res.boxes
        for box in boxes:
            # границы
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # вероятность
            conf = math.ceil((box.conf[0] * 100)) / 100
            # имя класса
            cls = int(box.cls[0])

            # определять только нужные категории
            current_class = class_names[cls]

            if current_class in ('car', 'truck', 'bus', 'motorbike') and conf > 0.3:
                # cvzone.putTextRect(img, f'{class_names[cls]} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5)
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    results_tracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, ids = map(int, result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{ids}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # пересечение детекции и линии
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if ids not in total_count:
                total_count.append(ids)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Count: {len(total_count)}', (50, 50))

    cv2.imshow('Image', img)
    # cv2.imshow('ImageRegion', img_region)
    cv2.waitKey(1)
