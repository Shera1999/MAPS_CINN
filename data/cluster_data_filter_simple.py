# ── cluster_data_filter_simple.py ──
#!/usr/bin/env python3
"""
Load your raw unobservables CSV, drop any row with NaNs in the five
target columns, scale Y, and write out processed_data/.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

TARGET_COLS = [
    'centralCoolingTime',
    'centralNumDens',
    'slopeNumDens',
    'concentrationPhys',
    'concentrationScaled',
    'centralEntropy',
    'bcg_SubhaloBHMass',
    'bcg_SubhaloBHMdot',
    'bcg_SubhaloSFR',
    'bcg_SubhaloSFRinHalfRad',
    'bcg_SubhaloMass',
    'bcg_StellarMass',
    'bcg_mass_ratio',
    'Group_M_Crit500',
    'Group_R_Crit500',
    'GroupGasMetallicity',
    'GroupStarMetallicity',
    'GroupSFR',
    'GroupVel_magnitude',
    'GroupBHMass',
    'GroupBHMdot',
    'Offset_magnitude',
    'GasMetalFrac_H',
    'GasMetalFrac_He',
    'GasMetalFrac_total',
    'lookback_time_Gyr',
    'GroupGasMass',
    'last_T_coll',
    'mean_T_coll',
    'last_V_coll',
    'mean_V_coll',
    'last_M_Crit500_coll',
    'mean_M_Crit500_coll',
    'last_Subcluster_mass',
    'mean_Subcluster_mass',
    'last_Mass_ratio',
    'mean_Mass_ratio',
    'last_d_peri',
    'mean_d_peri'
]

def generate_processed_data(unobs_csv, out_dir="processed_data"):
    # 1) Load
    df = pd.read_csv(unobs_csv)

    # 2) Drop any rows with missing targets
    before = len(df)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before-after}/{before} rows with missing targets.")

    # 3) Extract Y and meta
    Y_raw = df[TARGET_COLS].to_numpy()
    meta  = df[['HaloID','Snapshot']]

    # 3.5) Identify any non-finite or infinite values
    fin_mask = np.isfinite(Y_raw)
    if not fin_mask.all():
        bad_locs = np.argwhere(~fin_mask)
        print("Found non-finite entries in your target data:")
        for row_idx, col_idx in bad_locs:
            orig_index = df.index[row_idx]
            col_name   = TARGET_COLS[col_idx]
            bad_val    = Y_raw[row_idx, col_idx]
            print(f"  • filtered-df row {row_idx} (orig CSV idx {orig_index}), "
                  f"column '{col_name}': {bad_val!r}")
        raise ValueError("Aborting: please clean or filter out these infinities first.")

    # 4) Fit + apply target scaler
    tar_sc   = StandardScaler().fit(Y_raw)
    Y_scaled = tar_sc.transform(Y_raw)

    # 5) Write out
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(Y_scaled, columns=TARGET_COLS) \
      .to_csv(f"{out_dir}/Y.csv", index=False)
    meta.to_csv(f"{out_dir}/meta.csv", index=False)
    joblib.dump(tar_sc, f"{out_dir}/tar_scaler.pkl")
    print(f"Processed targets saved to '{out_dir}/'")

if __name__ == "__main__":
    import argparse

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
         description="Clean & scale raw unobservables CSV"
    )
    p.add_argument(
         "--unobs_csv",
         type=str,
         default="data/processed_features.csv",
         help="Path to your raw CSV of cluster features"
     )
    p.add_argument(
         "--out_dir",
         type=str,
         default="processed_data",
         help="Where to write Y.csv, meta.csv, tar_scaler.pkl"
     )
    args = p.parse_args()

    generate_processed_data(
        unobs_csv = args.unobs_csv,
         out_dir   = args.out_dir
     )
