# inspect_missing_features.py
import pandas as pd

# load original raw CSV of unobservables
df_raw = pd.read_csv("data/processed_features.csv")

# parse keys
df_raw["key"] = df_raw.apply(
    lambda r: f"{int(r.HaloID)}_{int(r.Snapshot)}", axis=1
)

# read missing keys
with open("missing_images.txt") as f:
    names = [line.strip() for line in f]
miss_keys = set(nm.replace(".jpg","").split("_halo_")[1].rsplit("_proj_",1)[0]
                for nm in names)

# filter raw
df_miss = df_raw[df_raw["key"].isin(miss_keys)]
print(df_miss)
df_miss.to_csv("eliminated_clusters.csv", index=False)
print("Wrote eliminated_clusters.csv for manual review")
