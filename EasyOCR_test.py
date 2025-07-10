import os
import cv2
import easyocr
import pandas as pd
from difflib import SequenceMatcher

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Paths
image_dir = 'OCR_dataset/test/cropped_images'
label_dir = 'OCR_dataset/test/labels'

results = []

# Process each cropped image
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

    if not os.path.exists(label_path):
        print(f"❌ Label missing for {filename}")
        continue

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Could not read {filename}")
        continue

    # Read ground truth text (extract last column as plate string)
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        gt_text = line.split("\t")[-1].upper()

    # Run OCR on the cropped plate image
    ocr_result = reader.readtext(image)
    pred_text = ocr_result[0][1].replace(" ", "").upper() if ocr_result else ""

    # Evaluate result
    similarity = SequenceMatcher(None, pred_text, gt_text).ratio()
    is_exact = pred_text == gt_text

    results.append({
        "filename": filename,
        "ground_truth": gt_text,
        "prediction": pred_text,
        "similarity": round(similarity, 3),
        "exact_match": is_exact
    })

# Save and display results
df = pd.DataFrame(results)

print("\n📊 Evaluation Complete. Sample Results:\n")
print(df.head(30))

# Optionally: print overall accuracy
accuracy = df["exact_match"].mean()
print(f"\n✅ Exact Match Accuracy: {accuracy:.3f}")
