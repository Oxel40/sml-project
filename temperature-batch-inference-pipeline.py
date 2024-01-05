import os


def main():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from datetime import datetime
    import dataframe_image as dfi
    import matplotlib.pyplot as plt
    from numpy.lib.stride_tricks import sliding_window_view
    from sklearn.metrics import mean_squared_error

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("temp_model", version=7)
    model_dir = model.download()
    model = joblib.load(model_dir + "/temp_model.pkl")
    scaler = mr.get_model("temp_scaler", version=3)
    scaler_dir = scaler.download()
    scaler = joblib.load(scaler_dir + "/temp_scaler.pkl")

    # Format data
    window_size = 24
    duration = 24

    temp_fg = fs.get_feature_group(name="weather", version=3)
    temp_df = temp_fg.read()
    temp_df = temp_df.set_index("date_time")
    temp_df = temp_df.asfreq('h')
    temp_df = temp_df.drop(columns=["id"])

    window = sliding_window_view(scaler.transform(temp_df.values), (window_size, 1))
    data = window[:-1]
    target = window[1:, :, -1]

    data = data.reshape((data.shape[0], -1))[-duration:]
    target = target.reshape((target.shape[0], -1))[-duration:]
    
    # Inference
    pred = model.predict(data)
    temp = pred[:, 0]
    print("Temperature predicted:", temp)
    dataset_api = project.get_dataset_api()
   
    monitor_fg = fs.get_or_create_feature_group(name="weather_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Weather Temperature Prediction/Outcome Monitoring"
                                                )
    
    y_true = scaler.inverse_transform(target.reshape(pred.shape))
    y_pred = scaler.inverse_transform(pred)
    error_mse = mean_squared_error(
                    y_true = y_true[:, 0],
                    y_pred = y_pred[:, 0]
                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'mse': [error_mse],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    history_df["datetime"] = pd.to_datetime(history_df["datetime"])

    history_df = history_df.sort_values("datetime")
    history_df = history_df.set_index("datetime")

    print(history_df.info())
    print(history_df.head())

    fig, ax = plt.subplots(figsize=(15, 4))
    history_df["mse"].plot(ax=ax)
    ax.legend()
    fig.savefig("./recent_temp_prediction_mses.png")
    
    dataset_api.upload("./recent_temp_prediction_mses.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    main()

