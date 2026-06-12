# 🔬 Defect Inspector

An industrial surface defect detection system built with YOLOv11 and Streamlit.
Trained on the NEU Steel Surface Defect Dataset to detect 6 types of manufacturing defects in real time.

---

## 📸 What It Does

- Detects 6 defect classes on steel surfaces using YOLOv11m
- Displays bounding boxes with confidence scores on uploaded images
- Builds a spatial heatmap showing where defects appear most frequently
- Tracks defect statistics across multiple images in a session
- Exports detection history as CSV

---

## 🏷️ Defect Classes

| Class | Description |
|---|---|
| Crazing | Network of fine surface cracks |
| Inclusion | Foreign material embedded in surface |
| Patches | Irregular surface patches |
| Pitted Surface | Small pits or holes |
| Rolled-in Scale | Scale pressed into surface during rolling |
| Scratches | Linear surface scratches |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv11m | Object detection model |
| Ultralytics | YOLO training framework |
| Streamlit | Dashboard UI |
| OpenCV | Image processing & box drawing |
| Plotly | Interactive charts |
| PyTorch + CUDA | GPU accelerated training |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| mAP50 | 0.734 |
| mAP50-95 | 0.400 |
| Precision | 0.688 |
| Recall | 0.706 |

Trained on RTX 4050 6GB with:
- 1,800 images across 6 classes
- 150 epochs
- Freeze=10 (backbone frozen)
- AdamW optimizer with cosine LR decay
- Heavy augmentation pipeline

---

## 📁 Project Structure

    defect-inspector/
    ├── model/
    │   └── best.pt          ← trained YOLOv11m weights
    ├── utils/
    │   ├── __init__.py
    │   ├── detector.py      ← YOLO inference wrapper
    │   └── heatmap.py       ← spatial heatmap generator
    ├── app.py               ← Streamlit dashboard
    ├── train.py             ← training script
    └── README.md

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install ultralytics streamlit opencv-python numpy pandas plotly
```

### 2. Launch the dashboard
```bash
cd defect-inspector
python -m streamlit run app.py
```

### 3. Load the model
- Set model path to `model/best.pt`
- Click **⚡ Load Model**
- Upload any steel surface image

---

## 🗂️ Dataset

**NEU Surface Defect Dataset**
- 1,800 grayscale steel surface images
- 6 defect classes, 300 images per class
- Available on [Roboflow Universe](https://universe.roboflow.com)

---

## 📈 Dashboard Features

| Feature | Description |
|---|---|
| 🖼️ Detection view | Bounding boxes + labels + confidence |
| 🌡️ Spatial heatmap | Defect location accumulation map |
| 📊 Class frequency | Bar chart per defect type |
| 🥧 Distribution | Pie chart of defect proportions |
| 📈 Confidence scores | Strip plot per detection |
| 📋 Detection log | Full history exportable to CSV |

---

## 👤 Author

Built as a professional case study combining deep learning, computer vision, and data visualization.
