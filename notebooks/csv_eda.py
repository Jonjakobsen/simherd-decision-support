#%%
import pandas as pd
from utils import RAW_DATA_DIR


df = pd.read_csv(RAW_DATA_DIR / "herd_data_raw.csv")

print((df["is_sick"] == 0).sum())

print(df["is_sick"].value_counts())

sick_cows = df.loc[df["is_sick"] == 1, "cow_id"].unique()
print(sick_cows)

sick_days_per_cow = (
    df[df["is_sick"] == 1]
    .groupby("cow_id")
    .size()
    .rename("sick_days")
)

print(sick_days_per_cow)
print(sick_days_per_cow.unique())
# %%
