import os
from PIL import Image
import torch
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# 📦 Load TrOCR model & processor
model_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 16
model.config.eos_token_id = processor.tokenizer.sep_token_id

# 📁 Load your JSON dataset
data_files = {
    "train": "OCR_dataset/train/train_data.json",
    "validation": "OCR_dataset/valid/valid_data.json"
}
dataset = load_dataset("json", data_files=data_files)

# 🔄 Preprocessing: convert image & tokenize label
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    
    label_ids = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=16,
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze(0)
    label_ids[label_ids == processor.tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel, "labels": label_ids}


processed_dataset = dataset.map(preprocess, remove_columns=["image_path", "text"])

# 🧩 Custom data collator to handle image inputs
def custom_collate_fn(features):
    pixel_values = torch.stack([
        torch.tensor(f["pixel_values"]) if not isinstance(f["pixel_values"], torch.Tensor) else f["pixel_values"]
        for f in features
    ])
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(f["labels"]) if not isinstance(f["labels"], torch.Tensor) else f["labels"]
         for f in features],
        batch_first=True,
        padding_value=-100
    )
    return {"pixel_values": pixel_values, "labels": labels}



# ⚙️ Training settings
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=5e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

# 🏋️‍♂️ Setup Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=custom_collate_fn
)

# 🚀 Start training
trainer.train()

# 💾 Save model
trainer.save_model("./trocr_finetuned")
