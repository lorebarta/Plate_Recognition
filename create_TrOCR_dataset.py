import os
import cv2
import json

# Crop images and create the labels for the TROCR model

# Input paths
image_folder = "OCR_dataset/test/detected_plates"
label_folder = "OCR_dataset/test/labels"

# Output paths
output_folder = "OCR_dataset/test/cropped_images"
os.makedirs(output_folder, exist_ok=True)

json_data = []

for label_file in os.listdir(label_folder):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_folder, label_file)
    with open(label_path, "r") as f:
        line = f.readline().strip()
        parts = line.split("\t")
        if len(parts) != 6:
            continue

        image_name, x, y, w, h, plate_text = parts
        x, y, w, h = map(int, (x, y, w, h))
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Crop plate
        x1, y1, x2, y2 = x, y, x + w, y + h
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_filename = os.path.join(output_folder, image_name)
        cv2.imwrite(crop_filename, crop)

        # Append JSON entry
        json_data.append({
            "image_path": crop_filename,
            "text": plate_text.strip().upper()
        })

# Write JSON
with open("test_data.json", "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ Done. Cropped {len(json_data)} plates and saved to train_data.json")
