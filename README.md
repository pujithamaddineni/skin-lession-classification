🩺 Skin Lesion Classification using EfficientNetB0
📌 Overview

This project implements a binary image classification model to detect benign vs malignant skin lesions using the ISIC 2019 dataset. The model is built on EfficientNetB0 with transfer learning, trained in TensorFlow/Keras, and deployed with a simple UI (Streamlit/Flask).

The goal is to assist dermatologists and researchers in early skin cancer detection.

📂 Dataset

Source: ISIC 2019 Challenge Dataset

Classes:

Benign (non-cancerous)

Malignant (cancerous, e.g., melanoma)

Preprocessing:

Image resizing (224x224)

Normalization (0–1)

Data augmentation (rotation, flip, zoom)

🧠 Model

Base Model: EfficientNetB0 (pretrained on ImageNet)

Modifications:

Global Average Pooling

Dense layers with Dropout

Final Sigmoid layer for binary classification

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

⚙️ Installation
# Clone repository
git clone https://github.com/your-username/skin-lesion-classification.git
cd skin-lesion-classification

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

🚀 Usage
1️⃣ Train the Model
python train.py --epochs 20 --batch_size 32 --dataset ./data

2️⃣ Evaluate the Model
python evaluate.py --model checkpoints/best_model.h5 --test_data ./data/test


                precision    recall  f1-score   support

      Benign       0.94      0.88      0.91      4162
   Malignant       0.58      0.76      0.65       905

    accuracy                           0.86      5067
   macro avg       0.76      0.82      0.78      5067
weighted avg       0.88      0.86      0.86      5067

(Values are illustrative; update with your actual results.)

🔮 Future Improvements

Use EfficientNetV2 for better accuracy

Apply Grad-CAM for model explainability

Deploy as a web/mobile app

Integrate with clinical datasets for real-world validation
