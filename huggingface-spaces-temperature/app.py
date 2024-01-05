import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("temp_model", version=7)
model_dir = model.download()
model = joblib.load(model_dir + "/temp_model.pkl")
scaler = mr.get_model("temp_scaler", version=3)
scaler_dir = scaler.download()
scaler = joblib.load(scaler_dir + "/temp_scaler.pkl")
print("Model and Scaler downloaded")

def get_window(window_size):
    window_size = 24

    temp_fg = fs.get_feature_group(name="weather", version=3)
    temp_df = temp_fg.read(read_options={"use_hive": True})
    temp_df = temp_df.set_index("date_time")
    temp_df = temp_df.asfreq('h')
    temp_df = temp_df.drop(columns=["id"])

    return scaler.transform(temp_df.values[-window_size:]), temp_df.index.max()

def temp_prediction():
    window_size = 24
    duration = 24

    data, latest_timestamp = get_window(window_size)
    #runup = scaler.inverse_transform(data.copy())
    for _ in range(duration):
        inp = data[-window_size:].flatten('F').reshape((1, -1))
        res = model.predict(inp, verbose=0)
        data = np.concatenate((data, res), axis=0)
    print("data shape", data.shape)

    res = pd.DataFrame(scaler.inverse_transform(data[window_size:])[:, 0:1], columns=["Temperature"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    res["Temperature"].plot(ax=ax, label="Predicted Temperature")
    plt.xlabel(f"Hours from {latest_timestamp}")
    plt.ylabel("Temperature (*C)")
    plt.legend()
    return fig
        
demo = gr.Interface(
    fn=temp_prediction,
    title="Temperature Prediction for the comming 24h",
    description="Temperature Prediction for the comming 24h",
    allow_flagging="never",
    inputs=[],
    outputs=gr.Plot(label="Temperature (*C)")
)

demo.launch(debug=True)