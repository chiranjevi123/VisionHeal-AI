# VisionHeal AI: Deep Learning for Rural Diagnostics

## 🚀 The Vision
To provide a first-line diagnostic screening tool for Tuberculosis and Eye Diseases in areas with zero access to specialists.

## 🛠️ Technical Stack
- **Framework:** TensorFlow / Keras 
- **Architecture:** MobileNetV2 / ResNet50 (Transfer Learning) 
- **Explainability:** Grad-CAM Class Activation Maps 
- **Interface:** Streamlit Web UI 

## 🧠 Model Logic
1. **Input:** Chest X-ray or Retinal Fundus image[cite: 13].
2. **Analysis:** The CNN identifies morphological abnormalities (e.g., opacities for TB, hemorrhages for Eye disease).
3. **Transparency:** A Grad-CAM heatmap is overlaid to show exactly where the model "looked" to make its decision.
4. **Output:** Diagnostic probability + visual evidence for medical verification.
