import os
import subprocess
import pandas as pd
import hopsworks


def fetch_dataframe():
    subprocess.run(["./fetch"], cwd=os.path.join(os.getcwd(), "./data/")) 

    temp_df = None

    for fid in range(101):
        try:
            archive_df = pd.read_csv(f"./data/97100-{fid}-archive.csv", delimiter=';')
            latest_df = pd.read_csv(f"./data/97100-{fid}-latest.csv", delimiter=';')
        except FileNotFoundError:
            continue
        tmp_df = pd.concat([archive_df, latest_df])
        print(f"=== {fid} ===")
        print(tmp_df.info())
        tmp_df = tmp_df.drop(columns=["Kvalitet", "Unnamed: 4", "Tidsutsnitt:"])
        tmp_df["Date-Time"] = pd.to_datetime(tmp_df["Datum"] + " " + tmp_df["Tid (UTC)"])
        tmp_df = tmp_df.drop(columns=["Datum", "Tid (UTC)"])
        print(tmp_df.head())
        if temp_df is None:
            temp_df = tmp_df
        else:
            temp_df = pd.merge(temp_df, tmp_df, how="outer", on="Date-Time")

    temp_df = temp_df.set_index("Date-Time")

    temp_df = temp_df.sort_values("Date-Time")
    temp_df = temp_df.drop_duplicates()
    temp_df = temp_df.interpolate()
    temp_df = temp_df.dropna()
    temp_df = temp_df.reset_index()
    temp_df["ID"] = temp_df.index

    new_names = {}
    for name in temp_df.columns:
        tmp_name = name.lower().replace('-', '_').replace(' ', '_')
        tmp_name = tmp_name.encode('ascii',errors='ignore')
        new_names[name] = tmp_name.decode()
    temp_df = temp_df.rename(columns=new_names)
    temp_df.columns

    return temp_df

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    temp_fg = fs.get_feature_group(name="weather",version=2)
    temp_df = fetch_dataframe()
    print(temp_df.info())
    print(temp_df.head())
    temp_fg.insert(temp_df)


if __name__ == "__main__":
    main()