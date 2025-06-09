# 🩺 Pneumonia Detection using Deep Learning

This project focuses on detecting **Pneumonia** from chest X-ray images. It leverages **DenseNet121**, a deep learning model, implemented with TensorFlow and Keras. For easy interaction and deployment, a Gradio web interface is included.

---

## 📦 Dataset

* **Source**: [Kaggle Dataset - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Classes**:
    * `NORMAL`
    * `PNEUMONIA`
* **Dataset Split**: The dataset is organized into `train/`, `val/`, and `test/` directories.

---

## 🚀 Features

* ✅ **Transfer Learning**: Utilizes DenseNet121 pretrained on ImageNet for robust feature extraction.
* ✅ **Data Augmentation**: Enhances model generalization by artificially expanding the training dataset.
* ✅ **Regularization**: Incorporates Dropout and Batch Normalization layers to prevent overfitting.
* ✅ **Callbacks**: Employs EarlyStopping and ModelCheckpoint for efficient training and model saving.
* ✅ **Learning Rate Scheduler**: Dynamically adjusts the learning rate during training for optimal convergence.
* ✅ **Interactive Web Deployment**: Provides a user-friendly Gradio-based web interface.

---

## ⚙️ Installation

[Instructions for installation would go here. Please add detailed steps for setting up the environment, dependencies, etc.]

---

## 📚 Usage

### 1️⃣ Train the Model

The dataset is automatically downloaded via Kagglehub. The model is trained on chest X-ray images resized to 224x224 pixels. Training checkpoints are saved as `/content/pneumonia_detector.h5`.

### 2️⃣ Visualize Results

Matplotlib visualizations are included in the code to track training progress. For example, you can plot the accuracy over epochs:

```python
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
```
###🌐 Deployment
Launch an interactive web interface using Gradio for easy inference. This allows you to upload an X-ray image and get a pneumonia prediction.

To launch the demo, run the deployment script ( within a Jupyter notebook):
###🏗️ Model Architecture
The model architecture builds upon the DenseNet121 backbone, which is initialized with pre-trained ImageNet weights and kept frozen during initial training:

-DenseNet121 (imagenet weights, frozen)
--GlobalAveragePooling2D
---BatchNormalization
----Dropout(0.4)
-----Dense(256, activation='ReLU')
------BatchNormalization
-------Dropout(0.3)
--------Dense(1, activation='Sigmoid')
This architecture is designed to classify images into one of two classes: NORMAL or PNEUMONIA.
###💡 Future Improvements
-Hyperparameter Tuning: Further optimize learning rate, dropout rates, batch size, and other training parameters for potentially better performance and faster convergence.
-Grad-CAM Visualizations: Implement Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of the X-ray image the model focuses on when making a prediction, enhancing model interpretability and trust.
-Deployment on Mobile/Web: Explore deploying the trained model to mobile devices (e.g., using TensorFlow Lite) or integrate it into a more robust web application for wider accessibility and real-world use.
Larger Dataset: Experiment with larger and more diverse chest X-ray datasets to improve generalization and robustness.
###🙏 Acknowledgements
-Dataset: Heartfelt thanks to Paul Mooney for providing the comprehensive Chest X-Ray Images (Pneumonia) dataset on Kaggle.
-Model Backbone: Appreciation to the developers of DenseNet121 and the ImageNet dataset for providing a powerful pre-trained model for transfer learning.



