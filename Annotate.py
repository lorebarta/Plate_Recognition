import os
import cv2
from ultralytics import YOLO

### Annotate images using yolo model and seva them in the yolo_processed folder

# === Percorsi ===
image_dir = 'PlateDetection_dataset/train/images'
model_path = 'PlateDetection_model/yolov8xl/best_xl.pt'
output_dir = 'Annotated/train/extra_large'
os.makedirs(output_dir, exist_ok=True)

# === Carica il modello YOLO ===
model = YOLO(model_path)

# === Loop immagini ===
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Immagine non trovata: {filename}")
        continue

    print(f"📷 Elaborazione: {filename}")
    results = model(image_path, device='cpu')[0]

    H, W = image.shape[:2]
    boxes = results.boxes
    if boxes is None or boxes.xyxy.shape[0] == 0:
        print("❌ Nessuna targa rilevata.")
        continue

    for i, (box, conf) in enumerate(zip(boxes.xyxy, boxes.conf)):
        if float(conf) < 0.3:  # puoi aumentare se vuoi più precisione
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"plate ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === Salva immagine annotata ===
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

print(f"\n✅ Completato. Immagini salvate in: {output_dir}")
