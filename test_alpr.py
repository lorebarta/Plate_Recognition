import os
import cv2
from fast_alpr import ALPR

# Initialize FastALPR (pretrained models are downloaded on the first run)
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

input_dir  = os.path.join(os.path.dirname(__file__), "OCR_dataset_full", "test", "images")
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
            conf = ocr.confidence
            if isinstance(conf, list):  # fast-alpr >= 0.4: one value per character
                conf = sum(conf) / len(conf)
            print(f"  Plate: {ocr.text}   Confidence: {conf:.2f}")
        else:
            print("  OCR failed or returned empty result.")


    # draw on the image and save
    ann = alpr.draw_predictions(img).image  # fast-alpr >= 0.4 returns a result object
    cv2.imwrite(os.path.join(output_dir, fn), ann)
