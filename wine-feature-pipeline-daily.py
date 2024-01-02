import os


def generate_wine(quality,
        type,
        fixed_acidity_mean,
        fixed_acidity_std,
        volatile_acidity_mean,
        volatile_acidity_std,
        citric_acid_mean,
        citric_acid_std,
        residual_sugar_mean,
        residual_sugar_std,
        chlorides_mean,
        chlorides_std,
        free_sulfur_dioxide_mean,
        free_sulfur_dioxide_std,
        total_sulfur_dioxide_mean,
        total_sulfur_dioxide_std,
        density_mean,
        density_std,
        ph_mean,
        ph_std,
        sulphates_mean,
        sulphates_std,
        alcohol_mean,
        alcohol_std):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({'type': type,
                       'fixed_acidity': [random.gauss(fixed_acidity_mean, fixed_acidity_std)],
                       'volatile_acidity': [random.gauss(volatile_acidity_mean, volatile_acidity_std)],
                       'citric_acid': [random.gauss(citric_acid_mean, citric_acid_std)],
                       'residual_sugar': [random.gauss(residual_sugar_mean, residual_sugar_std)],
                       'chlorides': [random.gauss(chlorides_mean, chlorides_std)],
                       'free_sulfur_dioxide': [random.gauss(free_sulfur_dioxide_mean, free_sulfur_dioxide_std)],
                       'total_sulfur_dioxide': [random.gauss(total_sulfur_dioxide_mean, total_sulfur_dioxide_std)],
                       'density': [random.gauss(density_mean, density_std)],
                       'ph': [random.gauss(ph_mean, ph_std)],
                       'sulphates': [random.gauss(sulphates_mean, sulphates_std)],
                       'alcohol': [random.gauss(alcohol_mean, alcohol_std)]
                      })
    df['quality'] = quality
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    # Data parsing and cleanup
    wine_df = pd.read_csv("./winequalityN.csv")
    wine_df = wine_df.dropna()
    wine_df['type'] = pd.get_dummies(wine_df['type'], dtype=int).drop('white',axis = 1)
    new_names = {}
    for name in wine_df.columns:
        new_names[name] = name.lower().replace(' ', '_')
    wine_df = wine_df.rename(columns=new_names)

    # Data generation
    quality = round(random.gauss(wine_df["quality"].mean(), wine_df["quality"].std()))

    sub_df = wine_df.query("quality == @quality")
    type = int(random.random() < sub_df["type"].mean())

    wine_df = generate_wine(quality,
                            type,
                            sub_df["fixed_acidity"].mean(),        sub_df["fixed_acidity"].std(),
                            sub_df["volatile_acidity"].mean(),     sub_df["volatile_acidity"].std(),
                            sub_df["citric_acid"].mean(),          sub_df["citric_acid"].std(),
                            sub_df["residual_sugar"].mean(),       sub_df["residual_sugar"].std(),
                            sub_df["chlorides"].mean(),            sub_df["chlorides"].std(),
                            sub_df["free_sulfur_dioxide"].mean(),  sub_df["free_sulfur_dioxide"].std(),
                            sub_df["total_sulfur_dioxide"].mean(), sub_df["total_sulfur_dioxide"].std(),
                            sub_df["density"].mean(),              sub_df["density"].std(),
                            sub_df["ph"].mean(),                   sub_df["ph"].std(),
                            sub_df["sulphates"].mean(),            sub_df["sulphates"].std(),
                            sub_df["alcohol"].mean(),              sub_df["alcohol"].std())

    return wine_df


def main():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    for _ in range(5):
        wine_fg.insert(get_random_wine())


if __name__ == "__main__":
    main()