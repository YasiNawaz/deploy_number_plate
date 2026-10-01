from ultralytics import YOLO
import gradio as gr 


import os
from ultralytics import YOLO

model_path = os.path.join(os.path.dirname(__file__), "best (1).pt")

model = YOLO(model_path)    

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = 'image', outputs = "image" )
app.launch(
    server_name="0.0.0.0",
    server_port=int(__import__("os").environ.get("PORT", 10000))
)

# download libraries ---> pip install -r requirements.txt
# break the terminal   ---- ctrl +c
# run the command that opeen broeswe
#  python app.py
