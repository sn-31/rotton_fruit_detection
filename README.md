# 🍎 Fruit Freshness Detection Project

A Deep Learning project to classify fruits (Apples, Bananas, Oranges) as Fresh or Rotten. Built with TensorFlow and Gradio.

## 📂 Project Structure
- `train.py`: Script to download the dataset and train the CNN model.
- `app.py`: Gradio web interface for real-time predictions.
- `fruit_model.keras`: The trained model file (generated after running `train.py`).

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install tensorflow gradio kagglehub pillow numpy
```

### 2. Train the Model
If you don't have the `.keras` file yet, run the training script:
```bash
python train.py
```
*Note: This will download the dataset from Kaggle and train for 5 epochs.*

### 3. Launch the App
Run the Gradio interface:
```bash
python app.py
```
This will open a tab in your browser where you can upload images or use your webcam!

## 🛠️ Features
- **AI Classification**: Detects 6 categories of fruit states.
- **Webcam Support**: Capture photos directly from the browser.
- **Confidence Check**: Alerts you if the fruit is unknown or the image is unclear.
- **Modern UI**: Clean, responsive interface using Gradio Blocks.
