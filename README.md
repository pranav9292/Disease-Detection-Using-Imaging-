# Paddy Leaf Disease Prediction Using Thermal Images

## Overview
This project uses **deep learning (CNN)** to classify thermal images of paddy leaves into **diseased** and **healthy** categories. The dataset contains thermal images of paddy leaves captured with a FLIR E8 thermal camera.  

The model can predict the following classes:  

- Bacterial leaf blight (BLB)  
- Blast  
- Leaf spot  
- Leaf folder  
- Hispa  
- Healthy leaves  

---

## Dataset
- Source: [Kaggle – Thermal Images of Diseased & Healthy Leaves - Paddy](https://www.kaggle.com/sujaradha/thermal-images-diseased-healthy-leaves-paddy)  
- Number of images: 636  
- Class distribution:

| Class                   | Images |
|-------------------------|-------|
| Bacterial leaf blight   | 220   |
| Blast                   | 67    |
| Leaf spot               | 80    |
| Leaf folder             | 34    |
| Hispa                   | 142   |
| Healthy leaves          | 93    |

- Images are stored in folders corresponding to each class.  

---

## Project Structure

```
SAFAS/
│
├─ models/ # Saved Keras model
├─ thermal images UL/ # Dataset folder (class-wise)
│ ├─ BLB/
│ ├─ Blast/
│ ├─ healthy/
│ ├─ hispa/
│ ├─ leaf folder/
│ └─ leaf spot/
├─ predictor.py # Main training & prediction script
├─ requirements.txt
└─ README.md
```

---

## Installation

1. Clone the repository:
    git clone https://github.com/pranav9292/SAFAS.git
    cd SAFAS

2. Create a virtual environment (recommended):
    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

3. Install dependencies:
    pip install -r requirements.txt


---

## Usage

1️⃣ Training the Model

    python predictor.py


2️⃣ Predict a New Image

    from predictor import predict_image

    predict_image(r"thermal images UL\Blast\Thermalimage1a.jpg")
