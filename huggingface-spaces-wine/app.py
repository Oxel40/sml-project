import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

columns = ['type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',
       'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
       'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']

def wine(type, fixed_acidity, volatile_acidity, citric_acid,
       residual_sugar, chlorides, free_sulfur_dioxide,
       total_sulfur_dioxide, density, ph, sulphates, alcohol):
    print("Calling function")
    df = pd.DataFrame([[type, fixed_acidity, volatile_acidity, citric_acid,
       residual_sugar, chlorides, free_sulfur_dioxide,
       total_sulfur_dioxide, density, ph, sulphates, alcohol]], 
                      columns=columns)
    print("Predicting")
    print(df)

    res = model.predict(df)

    print(res)        
    return res
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Analytics",
    description="Experiment with wine composition to predict quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(choices=['White', 'Red'], label="Type", type='index'),
        gr.inputs.Number(default=7.216579, label="Fixed Acidity"),
        gr.inputs.Number(default=0.339691, label="Volatile Acidity"),
        gr.inputs.Number(default=0.318722, label="Citric Acid"),
        gr.inputs.Number(default=5.444326, label="Residual Sugar"),
        gr.inputs.Number(default=0.056042, label="Chlorides"),
        gr.inputs.Number(default=30.525319, label="Free Sulfur Dioxide"),
        gr.inputs.Number(default=115.744574, label="Total Sulfur Dioxide"),
        gr.inputs.Number(default=0.994697, label="Density"),
        gr.inputs.Number(default=3.218395, label="pH"),
        gr.inputs.Number(default=0.531215, label="Sulphates"),
        gr.inputs.Number(default=10.491801, label="Alcohol"),
        ],
    outputs=gr.Number(label="Quality"))

demo.launch(debug=True)

