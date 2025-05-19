# debug_loader.py

import yaml
import torch
from data.combined_loader import CombinedDataset
from data.data_augmentor   import DataAugmentor
from torch.utils.data      import DataLoader

def inspect_batch(batch):
    (x0, x1), y = batch
    print("=== Batch shapes & NaN/Inf counts ===")
    for name, t in [("x0", x0), ("x1", x1), ("y", y)]:
        print(f"{name:2s}: shape={tuple(t.shape)}, "
              f"  NaNs={int(torch.isnan(t).sum())}, "
              f"  Infs={int(torch.isinf(t).sum())}")

def main():
    # 1) load dataset config
    dataset_cfg = yaml.safe_load(open("configs/dataset_config.yaml"))["dataset"]
    # 2) build the SSL augmentor
    augmentor = DataAugmentor(
        augmentations_config="configs/augmentations_config.yaml",
        dataset_config="configs/dataset_config.yaml",
        main_config="configs/main_config.yaml",
    )
    transform = augmentor.get_train_transform()
    # 3) load one small batch
    ds = CombinedDataset(dataset_cfg, transform)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    inspect_batch(batch)

if __name__ == "__main__":
    main()
