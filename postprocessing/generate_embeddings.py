#!/usr/bin/env python

import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from data.combined_loader import CombinedDataset
from models.joint_model    import JointModel
from data.data_augmentor   import DataAugmentor

def main():
    # 1) Load configs
    main_cfg    = yaml.safe_load(open("configs/main_config.yaml"))
    dataset_cfg = yaml.safe_load(open("configs/dataset_config.yaml"))["dataset"]
    model_cfg   = yaml.safe_load(open("configs/model_config.yaml"))["model"]

    # 2) SSL & cINN params must match train.py
    selected_ssl = model_cfg["selected_model"]
    ssl_cfg      = model_cfg[selected_ssl]
    ssl_lr       = float(ssl_cfg.get("learning_rate", 0.001))
    max_epochs   = int(main_cfg["training"]["max_epochs"])
    ssl_params   = {"learning_rate": ssl_lr, "max_epochs": max_epochs}

    cinn_cfg = model_cfg["cinn"]
    y_dim = pd.read_csv(f"{dataset_cfg['processed_data_dir']}/Y.csv").shape[1]
    if "input_dim" in ssl_cfg:
        x_dim = int(ssl_cfg["input_dim"])
    else:
        x_dim = int(ssl_cfg["hidden_dim"])
    cinn_params = {
        "learning_rate": float(cinn_cfg["learning_rate"]),
        "hidden_dim":    int(cinn_cfg["hidden_dim"]),
        "n_blocks":      int(cinn_cfg["n_blocks"]),
        "clamp":         float(cinn_cfg["clamp"]),
        "y_dim":         y_dim,
        "x_dim":         x_dim,
    }

    # 3) Load checkpoint
    ckpt_path = "checkpoints/joint_last.ckpt"
    model = JointModel.load_from_checkpoint(
        ckpt_path,
        selected_ssl=selected_ssl,
        ssl_params=ssl_params,
        cinn_params=cinn_params,
        λ_contrast=1.0,
        λ_cinn=1.0
    )
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 4) Test transform: duplicate single view into two
    augmentor = DataAugmentor(
        augmentations_config="configs/augmentations_config.yaml",
        dataset_config="configs/dataset_config.yaml",
        main_config="configs/main_config.yaml",
    )
    single_tf = augmentor.get_test_transform()
    def test_transform(img):
        x = single_tf(img)
        return x, x

    # 5) Dataset & DataLoader
    ds = CombinedDataset(dataset_cfg, test_transform)
    dl = DataLoader(
        ds,
        batch_size=main_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=main_cfg["training"]["num_workers"]
    )

    # 6) Extract & save embeddings
    all_embs = []
    with torch.no_grad():
        for (x0, _), _ in dl:
            x0 = x0.to(model.device)
            h  = model.backbone(x0).flatten(1)
            z  = model.projection_head(h)
            all_embs.append(z.cpu().numpy())

    embs = np.vstack(all_embs)
    embs = normalize(embs, axis=1)
    np.save("embeddings.npy", embs)
    print(f"✔️  Saved embeddings.npy with shape {embs.shape}")

if __name__ == "__main__":
    main()
