import os
import pandas as pd
import torch
from PIL import Image, ImageTk, ImageEnhance, ImageDraw
from difflib import SequenceMatcher
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.root.geometry("800x700")

        self.canvas_width = 600

        # Load TrOCR model
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "OCR_model/TrOCR_base_printed/checkpoint-462")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device).eval()

        # Load YOLO model
        self.yolo_model = YOLO("PlateDetection_model/runs_XL/extra_large_net/weights/best.pt")

        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.scrollable_frame = scrollable_frame

        top_frame = tk.Frame(scrollable_frame, width=self.canvas_width, height=60, bg="#cccccc")
        top_frame.pack(pady=10)
        top_frame.pack_propagate(0)

        load_btn = tk.Button(top_frame, text="Load Image", command=self.load_image)
        load_btn.pack(expand=True)

        self.canvas = tk.Label(scrollable_frame)
        self.canvas.pack(pady=10)

        tk.Label(scrollable_frame, text="Brightness").pack()
        self.brightness_slider = tk.Scale(scrollable_frame, from_=0.5, to=2.0, resolution=0.1,
                                          orient='horizontal', command=self.update_preview, length=self.canvas_width)
        self.brightness_slider.set(1.0)
        self.brightness_slider.pack(pady=5)

        tk.Label(scrollable_frame, text="Contrast").pack()
        self.contrast_slider = tk.Scale(scrollable_frame, from_=0.5, to=2.0, resolution=0.1,
                                        orient='horizontal', command=self.update_preview, length=self.canvas_width)
        self.contrast_slider.set(1.0)
        self.contrast_slider.pack(pady=5)

        tk.Label(scrollable_frame, text="Noise Intensity").pack()
        self.noise_slider = tk.Scale(scrollable_frame, from_=0, to=100, resolution=1,
                                     orient='horizontal', command=self.update_preview, length=self.canvas_width)
        self.noise_slider.set(0)
        self.noise_slider.pack(pady=5)

        btn_frame = tk.Frame(scrollable_frame)
        btn_frame.pack(pady=20)

        apply_btn = tk.Button(btn_frame, text="Apply to All Images", command=self.apply_to_folder)
        apply_btn.pack(side="left", padx=5)

        self.original_img = None
        self.display_img = None
        self.loaded_image_path = None

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.loaded_image_path = path
            img = Image.open(path)
            self.original_img = self.resize_image(img)
            self.update_preview()

    def resize_image(self, image):
        ratio = self.canvas_width / image.width
        new_height = int(image.height * ratio)
        return image.resize((self.canvas_width, new_height), Image.LANCZOS)

    def apply_edits(self, img):
        img = ImageEnhance.Brightness(img).enhance(self.brightness_slider.get())
        img = ImageEnhance.Contrast(img).enhance(self.contrast_slider.get())
        img_array = np.array(img)
        noise_level = self.noise_slider.get()
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def update_preview(self, event=None):
        if self.original_img:
            img = self.original_img.copy()
            img = self.apply_edits(img)
            self.display_img = img
            tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(image=tk_img)
            self.canvas.image = tk_img

    def apply_to_folder(self):
        image_dir = os.path.join("dataset", "test", "images")
        label_dir = os.path.join("dataset", "test", "labels")
        modified_dir = os.path.join(image_dir, "modified")
        annotated_dir = os.path.join(image_dir, "modified_annotated")
        cropped_dir = os.path.join(image_dir, "cropped_images")

        os.makedirs(modified_dir, exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)

        valid_ext = ('.jpg', '.jpeg', '.png')

        results = []

        for filename in os.listdir(image_dir):
            if not filename.lower().endswith(valid_ext):
                continue

            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            edited = self.apply_edits(img)

            modified_path = os.path.join(modified_dir, filename)
            edited.save(modified_path)

            img_np = np.array(edited)

            yolo_result = self.yolo_model(img_np, device="cpu")[0]
            boxes = yolo_result.boxes

            if boxes is None or boxes.xyxy.shape[0] == 0:
                print(f"No detections in {filename}")
                continue

            annotated = edited.copy()
            draw = ImageDraw.Draw(annotated)
            cropped_saved = False

            for box_tensor, conf_tensor in zip(boxes.xyxy, boxes.conf):
                conf = float(conf_tensor)
                if conf < 0.3:
                    continue

                x1, y1, x2, y2 = map(int, box_tensor.tolist())
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 15), f"{conf:.2f}", fill="red")

                if not cropped_saved:
                    cropped = edited.crop((x1, y1, x2, y2))
                    cropped_path = os.path.join(cropped_dir, filename)
                    cropped.save(cropped_path)
                    cropped_saved = True

            annotated_path = os.path.join(annotated_dir, filename)
            annotated.save(annotated_path)

            if not cropped_saved:
                print(f"No high-confidence detection in {filename}")
                continue

            # TrOCR OCR on cropped image
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                    gt_text = lines[1].strip().upper() if len(lines) >= 2 else ""

                pixel_values = self.processor(images=cropped, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                pred_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].upper()

                similarity = SequenceMatcher(None, pred_text, gt_text).ratio()
                is_exact = pred_text == gt_text

                results.append({
                    "filename": filename,
                    "ground_truth": gt_text,
                    "prediction": pred_text,
                    "similarity": round(similarity, 3),
                    "exact_match": is_exact
                })

        df = pd.DataFrame(results)
        print("\nOCR Evaluation Results:\n")
        print(df)

        if not df.empty:
            accuracy = df["exact_match"].mean()
            print(f"\nExact Match Accuracy: {accuracy:.3f}")
            self.plot_results(df)

    def plot_results(self, df):
        new_window = tk.Toplevel(self.root)
        new_window.title("OCR Results")
        new_window.geometry("800x600")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Exact Match Accuracy"], [df["exact_match"].mean()])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("TrOCR Model Accuracy")

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.resizable(False, False)
    root.mainloop()
