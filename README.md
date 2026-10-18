# Plate_Recognition

Riconoscimento automatico di targhe (ALPR) in due fasi:

1. **Detection**: un modello YOLOv8 trova la targa nell'immagine.
2. **OCR**: la targa ritagliata viene letta da un modello TrOCR (Microsoft)
   addestrato sulle targhe; come confronto c'è EasyOCR, usato senza addestramento.

I modelli sono stati addestrati su Google Colab (notebook in
`PlateDetection_training/` e `OCR_training/`). **I pesi addestrati non sono nella
repo**: senza di loro funzionano solo la demo con modelli pre-addestrati e il test
di EasyOCR (vedi sotto).

## Installazione

Serve Python 3.10+. La prima volta:

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # PyTorch solo CPU (più leggero)
pip install -r requirements.txt
```

Se hai una GPU NVIDIA, installa invece PyTorch con CUDA seguendo <https://pytorch.org/get-started/locally/>.

## Cosa si può lanciare subito

Al primo avvio questi script scaricano da soli i modelli pre-addestrati.

```bash
python test_alpr.py      # detection + OCR con fast-alpr sulle 30 immagini di test
                         # → stampa le targhe lette e salva le immagini annotate in output/
python EasyOCR_test.py   # EasyOCR sulle targhe già ritagliate, confrontato con le etichette
                         # → tabella dei risultati e accuratezza (circa 7%: per questo serve TrOCR)
```

## Con i modelli addestrati in proprio

1. **Detection (YOLOv8)**: apri `PlateDetection_training/train.ipynb` su Colab.
   Il dataset è [License Plate Recognition di Roboflow](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)
   (configurazione in `PlateDetection_training/data.yaml`). Copia il `best.pt`
   ottenuto in `PlateDetection_model/runs_XL/extra_large_net/weights/best.pt`
   (lo usa `gui.py`) oppure in `PlateDetection_model/yolov8xl/best_xl.pt` (lo usa `Annotate.py`).
2. **OCR (TrOCR)**: crea i ritagli con `python create_TrOCR_dataset.py`, poi addestra
   con `OCR_training/train.ipynb` (oppure `OCR_training/TrOCR_train.py`). Copia il
   checkpoint in `OCR_model/TrOCR_base_printed/checkpoint-462`.

Poi:

```bash
python trOCR_test.py     # accuratezza di TrOCR sulle targhe di test
python Annotate.py       # disegna i box YOLO sulle immagini
python gui.py            # interfaccia Tkinter per testare la robustezza: carichi un'immagine,
                         # regoli luminosità, contrasto e rumore, e con "Apply to All Images"
                         # rilanci detection + OCR su tutto il set di test
```

## Struttura

```
OCR_dataset_full/test/       30 immagini di test, etichette, targhe rilevate e ritagliate
PlateDetection_dataset/      un'immagine di esempio nel formato YOLO
PlateDetection_training/     notebook di training YOLOv8 + data.yaml
OCR_training/                notebook e script di fine-tuning TrOCR
PlateDetection_model/        qui vanno i pesi YOLO (non versionati)
OCR_model/                   qui va il checkpoint TrOCR (non versionato)
```
