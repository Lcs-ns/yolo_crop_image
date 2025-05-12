from ultralytics import YOLO
import cv2
import os

model_path = 'runs/detect/train/weights/best.pt'

input_folder = 'datasets/valid/images'
output_folder = 'placas_recortadas'

os.makedirs(output_folder, exist_ok=True)

model = YOLO(model_path)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)

    results = model(image_path)[0]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        placa_crop = img[y1:y2, x1:x2]

        output_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_placa{i}.jpg')
        cv2.imwrite(output_path, placa_crop)

    print(f"Processado: {filename}")

print("Extração de placas concluída.")
