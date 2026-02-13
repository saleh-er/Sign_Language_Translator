# 🤟 Real-Time Sign Language Translator

An AI-powered application that translates hand gestures into text in real-time. By mapping 21 hand landmarks, this project provides a lightweight and efficient alternative to heavy image-processing models.

## 🚀 Features
* **Real-time Detection:** High-speed tracking using MediaPipe.
* **Coordinate-Based Learning:** Uses geometric points rather than raw pixels, making it 10x faster and more accurate across different lighting.
* **Custom Dataset:** Includes a data collection pipeline to teach the AI any sign you choose.

## 🛠️ Tech Stack
* **Python 3.11**
* **MediaPipe** (Landmark extraction)
* **TensorFlow/Keras** (Neural Network)
* **OpenCV** (Computer Vision)
* **Scikit-Learn** (Data processing)

## 📦 Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/Sign-Language-Translator.git](https://github.com/yourusername/Sign-Language-Translator.git)