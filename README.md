🩺 Skin Lesion Classification using EfficientNetB0






📌 Overview

This project implements a binary image classification model to detect benign vs malignant skin lesions using the ISIC 2019 dataset.

✅ Model: EfficientNetB0 with transfer learning

✅ Framework: TensorFlow/Keras

✅ Deployment: Streamlit/Flask UI

🎯 Goal: To support early skin cancer detection for dermatologists & researchers

📂 Dataset

Source: ISIC 2019 Challenge Dataset

Classes:

🟢 Benign (non-cancerous)

🔴 Malignant (cancerous, e.g., melanoma)

Preprocessing steps:

Resize images → 224x224

Normalize pixel values → [0–1]

Apply data augmentation → rotation, flipping, zoom

🧠 Model Architecture

Base: EfficientNetB0 (ImageNet pretrained)

Custom layers:

Global Average Pooling

Dense layer(s) + Dropout

Sigmoid activation for binary classification

Training setup:

Loss: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

⚙️ Installation
# Clone repository
git clone https://github.com/your-username/skin-lesion-classification.git
cd skin-lesion-classification

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

🚀 Usage
🔹 Train the Model
python train.py --epochs 20 --batch_size 32 --dataset ./data

🔹 Evaluate the Model
python evaluate.py --model checkpoints/best_model.h5 --test_data ./data/test



➡ Upload a lesion image → Get prediction (Benign / Malignant)

📊 Results
Class	Precision	Recall	F1-Score	Support
Benign	0.94	0.88	0.91	4162
Malignant	0.58	0.76	0.65	905

Accuracy: 0.86

Macro Avg: Precision 0.76 | Recall 0.82 | F1-Score 0.78

Weighted Avg: Precision 0.88 | Recall 0.86 | F1-Score 0.86



🔮 Future Improvements

Upgrade to EfficientNetV2 / Vision Transformers (ViT)

Apply Grad-CAM for interpretability

Deploy as web/mobile app for clinicians

Fine-tune with larger, clinical datasets
