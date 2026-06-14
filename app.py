import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_model.keras")

# 1. Load the model
if not os.path.exists(MODEL_PATH):
    model = None
    print(f"Error: {MODEL_PATH} not found. Please run train.py first.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange',
               'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

def predict_fruit(img):
    if model is None:
        return "Model not found. Please run train.py to generate 'fruit_model.keras'."
    
    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    max_idx = np.argmax(prediction)
    confidence = prediction[max_idx]

    # Confidence Threshold (Unknown check)
    if confidence < 0.70:
        return {"Unknown Fruit (Low Confidence)": 1.0}
    
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}

# --- Custom UI with Blocks ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald")) as demo:
    gr.Markdown(
        """
        # Fresh and Rotton fruit detection 
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload or Capture Photo", sources=["upload", "webcam"], height=300)
            btn = gr.Button("Scan Fruit", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=3, label="Analysis Result")
            gr.Markdown("### How it works:")
            gr.Markdown("1. Upload/Capture a photo of an apple, banana, or orange.\n2. Click 'Scan Fruit'.\n3. The AI predicts freshness based on visual features.")

    btn.click(fn=predict_fruit, inputs=input_img, outputs=output_label)

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)
