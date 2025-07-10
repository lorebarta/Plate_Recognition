import os
import pandas as pd
import torch
from PIL import Image
from difflib import SequenceMatcher
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 🟢 Update this to your checkpoint folder
checkpoint_dir = "OCR_model/TrOCR_base_printed/checkpoint-462"
# checkpoint_dir = "OCR_model/TrOCR_base_stage1/checkpoint-XX"

# 🟢 Paths to test data
image_dir = "OCR_dataset_full/test/cropped_images"
label_dir = "OCR_dataset_full/test/labels"

# Load TrOCR model and processor
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

results = []

# Process each cropped image
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"❌ Label missing for {filename}")
        continue

    # Load ground truth (extract last column)
    with open(label_path, "r") as f:
        line = f.readline().strip()
        gt_text = line.split("\t")[-1].upper()

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Preprocess
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate prediction
    generated_ids = model.generate(pixel_values)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].upper()

    # Evaluate similarity
    similarity = SequenceMatcher(None, pred_text, gt_text).ratio()
    is_exact = pred_text == gt_text

    results.append({
        "filename": filename,
        "ground_truth": gt_text,
        "prediction": pred_text,
        "similarity": round(similarity, 3),
        "exact_match": is_exact
    })

# Convert results to DataFrame
df = pd.DataFrame(results)

# Display results
print("\n📊 Evaluation Complete. Sample Results:\n")
print(df.head(30))

# Optionally print overall accuracy
accuracy = df["exact_match"].mean()
print(f"\n✅ Exact Match Accuracy: {accuracy:.3f}")
