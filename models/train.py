# models/train.py
#!/usr/bin/env python

import yaml
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from data.combined_loader import CombinedDataset
from models.joint_model    import JointModel
from data.data_augmentor   import DataAugmentor

def main():
    # 1) Load configs
    main_cfg    = yaml.safe_load(open("configs/main_config.yaml"))
    dataset_cfg = yaml.safe_load(open("configs/dataset_config.yaml"))["dataset"]
    model_cfg   = yaml.safe_load(open("configs/model_config.yaml"))["model"]

    # 2) SSL params
    selected_ssl = model_cfg["selected_model"]
    ssl_cfg      = model_cfg[selected_ssl]
    ssl_lr       = float(ssl_cfg.get("learning_rate", 0.001))
    max_epochs   = int(main_cfg["training"]["max_epochs"])
    ssl_params   = {"learning_rate": ssl_lr, "max_epochs": max_epochs}

    # 3) cINN params
    cinn_cfg = model_cfg["cinn"]
    # y_dim from processed_data/Y.csv
    y_dim = pd.read_csv(f"{dataset_cfg['processed_data_dir']}/Y.csv").shape[1]
    # x_dim from SSL config: either input_dim or hidden_dim
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

    # 4) Build transforms
    augmentor = DataAugmentor(
        augmentations_config="configs/augmentations_config.yaml",
        dataset_config="configs/dataset_config.yaml",
        main_config="configs/main_config.yaml",
    )
    train_transform = augmentor.get_train_transform()

    # 5) DataLoader
    ds = CombinedDataset(dataset_cfg, train_transform)
    dl = DataLoader(
        ds,
        batch_size=main_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=main_cfg["training"]["num_workers"]
    )

    # 6) Model
    model = JointModel(
        selected_ssl=selected_ssl,
        ssl_params=ssl_params,
        cinn_params=cinn_params,
        λ_contrast=1.0,
        λ_cinn=1.0
    )

    # 7) Checkpoint callback
    ckpt_cb = ModelCheckpoint(
        monitor="loss_total",
        dirpath="checkpoints/",
        filename="joint-{epoch:02d}-{loss_total:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
    )

    # 8) Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if main_cfg["training"]["device"]=="gpu" else "cpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[ckpt_cb],
        log_every_n_steps=main_cfg["training"].get("log_every_n_steps", 10),
    )

    # 9) Fit & save
    trainer.fit(model, dl)
    trainer.save_checkpoint("checkpoints/joint_last.ckpt")
    print("✔️  Joint model checkpoint saved to checkpoints/joint_last.ckpt")

if __name__ == "__main__":
    main()
