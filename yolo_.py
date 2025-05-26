from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='datasets_2/data.yaml', epochs=50, imgsz=640)