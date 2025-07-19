
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Предобученная модель YOLOv8

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Запуск модели
    results = model(img)

    # Фильтруем только велосипеды (class 1 = bicycle в COCO)
    bicycles = [box for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls) if int(cls) == 1]

    # Отрисовка рамок только для велосипедов
    for box in bicycles:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Bicycle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Сохранение результата
    os.makedirs('static', exist_ok=True)
    cv2.imwrite('static/result.jpg', img)

    return jsonify({'count': len(bicycles)})

@app.route('/result')
def result_image():
    return send_file('static/result.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
