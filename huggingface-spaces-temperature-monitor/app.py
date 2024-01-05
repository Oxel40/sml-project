import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/df_recent.png")
dataset_api.download("Resources/images/recent_temp_prediction_mses.png")

with gr.Blocks() as demo:    
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction MSE History")
          input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Historical Prediction Performance (MSE)")
          input_img = gr.Image("recent_temp_prediction_mses.png", elem_id="mse-graph")        

demo.launch()
