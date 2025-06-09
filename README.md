# 🩺 Pneumonia Detection using Deep Learning

A deep learning project for detecting **Pneumonia** from chest X-ray images using **DenseNet121** with TensorFlow and Keras. Includes a Gradio web interface for easy deployment.

---

## 📦 Dataset

- **Source**: [Kaggle Dataset - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**:
  - `NORMAL`
  - `PNEUMONIA`
- Dataset Split:
  - `train/`
  - `val/`
  - `test/`

---

## 🚀 Features

- ✅ Transfer Learning with DenseNet121 (ImageNet pretrained)
- ✅ Data Augmentation for better generalization
- ✅ Dropout & BatchNormalization to prevent overfitting
- ✅ EarlyStopping & ModelCheckpoint
- ✅ Learning Rate Scheduler
- ✅ Interactive Gradio-based Web Deployment

---

## ⚙️ Installation

##📚 Usage
##1️⃣ Train the Model
-Dataset automatically downloaded via kagglehub

-Trained on chest X-ray images resized to 224x224

-Checkpoints saved as /content/pneumonia_detector.h5

##2️⃣ Visualize Results
# Matplotlib visualization included in code
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")

 ##Accuracy: ✅ 91.51%

##🌐 Deployment
Launch an interactive web interface using Gradio:
demo.launch(share=True)
🏗️ Model Architecture
DenseNet121 (imagenet weights, frozen)
└── GlobalAveragePooling2D
    └── BatchNormalization
        └── Dropout(0.4)
            └── Dense(256, ReLU)
                └── BatchNormalization
                    └── Dropout(0.3)
                        └── Dense(1, Sigmoid)

##💡 Future Improvements
-Hyperparameter Tuning (learning rate, dropout rates, etc.)

-Grad-CAM Visualizations for explainability

-Deployment on mobile/web (e.g., TFLite)

##🙏 Acknowledgements
-Dataset by Paul Mooney (Kaggle)

-Model backbone: DenseNet121 (pretrained on ImageNet)

