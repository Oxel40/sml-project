import os


def main():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from datetime import datetime
    import matplotlib.pyplot as plt
    from numpy.lib.stride_tricks import sliding_window_view

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("temp_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/temp_model.pkl")

    # Format data
    temp_fg = fs.get_feature_group(name="weather", version=3)
    temp_df = temp_fg.read()
    temp_df = temp_df.set_index("date_time")
    temp_df = temp_df.asfreq('h')
    temp_df = temp_df.drop(columns=["id"])

    window = sliding_window_view(temp_df.values, (12, 1))
    data = window[:-1]
    target = window[1:, :, -1]

    data = data.reshape((data.shape[0], -1))
    target = target.reshape((target.shape[0], -1))
    
    # Inference
    pred = model.predict(data[-24:])
    temp = pred[:, 0]
    print("Temperature predicted:", temp)
    dataset_api = project.get_dataset_api()
   
    #print(df)
    actual = temp_df.iloc[-24:]["lufttemperatur"]
    
    monitor_fg = fs.get_or_create_feature_group(name="weather_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Weather Temperature Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [temp],
        'actual': [actual],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(24*4)

    fig, ax = plt.subplots(figsize=(15, 4))
    df_recent.plot(ax)
    ax.legend()
    fig.savefig("./recent_temp_predictions.png")
    
    dataset_api.upload("./recent_temp_predictions.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    main()

