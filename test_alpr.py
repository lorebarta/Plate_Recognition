import os
import cv2
from fast_alpr import ALPR

# Initialize FastALPR
alpr = ALPR(
    detector_model="PlateDetection_model\\yolov8s\\best.onnxd",
    ocr_model="global-plates-mobile-vit-v2-model",
)

input_dir  = os.path.join(os.path.dirname(__file__), "yolo_processed")
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

for fn in os.listdir(input_dir):
    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_dir, fn)
    img  = cv2.imread(path)
    if img is None:
        print(f"Could not read {fn!r}")
        continue

    results = alpr.predict(img)
    print(f"\nImage: {fn}")
    if not results:
        print("  No plates detected.")
    for r in results:
        ocr = getattr(r, "ocr", None)
        if ocr and ocr.text.strip():
            print(f"  Plate: {ocr.text}   Confidence: {ocr.confidence:.2f}")
        else:
            print("  OCR failed or returned empty result.")


    # draw on the image and save
    ann = alpr.draw_predictions(img)
    cv2.imwrite(os.path.join(output_dir, fn), ann)
