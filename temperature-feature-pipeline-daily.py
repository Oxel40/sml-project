import os
import pandas as pd
import hopsworks

STATION_ID = 98230
PARAM = 1
URL = f"https://opendata-download.smhi.se/stream?type=metobs&parameterIds={PARAM}&stationId={STATION_ID}&period=latest-months"

def fetch_dataframe():
    temp_df = pd.read_csv(URL, delimiter=';', skiprows=9)

    temp_df = temp_df.drop(columns=["Unnamed: 4", "Tidsutsnitt:"])
    temp_df["Date-Time"] = pd.to_datetime(temp_df["Datum"] + " " + temp_df["Tid (UTC)"])
    temp_df = temp_df.drop(columns=["Datum", "Tid (UTC)"])
    temp_df.sort_values("Date-Time")
    temp_df.drop_duplicates()

    new_names = {}
    for name in temp_df.columns:
        new_names[name] = name.lower().replace('-', '_')
    temp_df = temp_df.rename(columns=new_names)

    return temp_df

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    temp_fg = fs.get_feature_group(name="weather",version=1)
    temp_df = fetch_dataframe()
    print(temp_df.info())
    print(temp_df.head())
    temp_fg.insert(temp_df)


if __name__ == "__main__":
    main()