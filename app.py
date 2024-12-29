import gradio as gr
from model import Trainer
import torch
import cv2
import tempfile
import numpy as np

def predict_beauty_score(img):
    trainer = Trainer()
    
    # Save the numpy array as an image temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Convert RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp.name, img_bgr)
        # Use the temporary file path
        image_tensor = trainer.image_to_tensor(tmp.name)
    
    prediction = trainer.predict(image_tensor)
    score = prediction.item() * 100
    
    # Decide which GIF to show based on the score
    if score < 20:
        gif_url = "https://i.pinimg.com/originals/9f/79/2a/9f792aed5881d409425de1a4361bc06b.gif"
    elif score < 40:
        gif_url = "https://i.pinimg.com/originals/ba/f5/c8/baf5c89c099b34decb7f4507b5144366.gif"
    elif score < 60:
        gif_url = "https://i.pinimg.com/originals/87/b6/dc/87b6dcfeec6f38a3836b1caf1d8fceab.gif"
    elif score < 80:
        gif_url = "https://i.pinimg.com/originals/3a/90/b8/3a90b87a337b79b9c8b7a3d9bf7250d7.gif"    
    else:
        gif_url = "https://i.pinimg.com/originals/f6/02/01/f6020120d9e99f7b106c557cdc1edb1f.gif"
    
    # Create formatted HTML outpu
    html_output = f"""
    <div style='text-align: center; padding: 20px;'>
        <h2 style='color: #FFFFFF; margin-bottom: 10px;'>Score</h2>
        <div style='font-size: 48px; font-weight: bold; color: #1a73e8;'>
            {score:.3f}
        </div>
        <br/>
        <div style='display: flex; justify-content: center;'>
            <img src="{gif_url}" alt="GIF" width="200" height="200" />
        </div>
        <p>Enjoy the fun! :)</p>
    </div>
    """
    
    return html_output

# Create Gradio interface
demo = gr.Interface(
    fn=predict_beauty_score,
    inputs=gr.Image(),  # Simple image input
    outputs=gr.HTML(),  # Using HTML output for custom formatting
    title="Image Beauty Score Predictor",
    description="""
    <ul>
        <li>Upload an image to get its beauty score prediction</li>
        <li>Please remember that this is just for fun and is not intended to downplay anybody and I, as the model creator believe everyone is beautiful the way it is</li>
        <li>Respect each other and most importantly have fun :)</li>
    </ul>
    """,
    examples=[
        ["6082308423334085331.jpg"],
        ["a00e1c819a87fa56bb1e6058d9814bae.jpg"]
    ],
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch()
