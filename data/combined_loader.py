# data/combined_loader.py

import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dataset_cfg, transform):
        """
        dataset_cfg: dict from configs/dataset_config.yaml under "dataset"
        transform: callable that takes a PIL image and returns (x0, x1)
        """
        img_dir       = dataset_cfg["data_path"]
        processed_dir = dataset_cfg["processed_data_dir"]

        # 1) Load scaled targets + meta
        dfY  = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
        meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"))

        # 2) Replicate each (halo, snapshot) 3× for proj=1,2,3
        projs = [1,2,3]
        meta_rep = pd.concat([meta.assign(proj=p) for p in projs], ignore_index=True)
        Y_rep    = dfY.values.repeat(len(projs), axis=0)

        # 3) Build key → Y lookup
        keys = [
            f"{int(r.HaloID)}_{int(r.Snapshot)}_{int(r.proj)}"
            for _, r in meta_rep.iterrows()
        ]
        self.Y_map = { keys[i]: Y_rep[i] for i in range(len(keys)) }

        # 4) List all JPEGs AND filter out those without a matching Y
        all_paths = list(Path(img_dir).glob("*.jpg"))
        kept, dropped = [], []
        for p in all_paths:
            key = self._normalize(p.name)
            if key in self.Y_map:
                kept.append(p)
            else:
                dropped.append(p.name)

        self.img_paths = kept
        if dropped:
            print(f"⚠️  Dropped {len(dropped)}/{len(all_paths)} images with no matching targets:")
            for fname in dropped[:20]:
                print(f"   - {fname}")
            if len(dropped) > 20:
                print(f"   ... plus {len(dropped)-20} more")
            # save full list for manual inspection
            with open("missing_images.txt", "w") as f:
                for fname in dropped:
                    f.write(fname + "\n")
            print("↪️  Wrote full list of missing images to 'missing_images.txt'")

        # 5) Store transform
        self.transform = transform

    def _normalize(self, fname: str) -> str:
        """
        From 'snap_050_halo_10003955_proj_2.jpg' → '10003955_50_2'
        """
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.match(r"^snap_(\d+)_halo_(\d+)_proj_(\d+)$", base)
        if not m:
            raise ValueError(f"Unexpected filename: {base}")
        snap, halo, proj = m.groups()
        return f"{int(halo)}_{int(snap)}_{int(proj)}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img  = Image.open(path).convert("RGB")
        x0, x1 = self.transform(img)         # two SSL views
        key     = self._normalize(path.name)
        y       = torch.from_numpy(self.Y_map[key].astype("float32"))
        return (x0, x1), y
