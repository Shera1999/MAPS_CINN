# models/joint_model.py
import pytorch_lightning as pl
import torch, torch.nn as nn
from models.cluster_cinn import cINN

from models.simclr  import SimCLRModel
from models.byol    import BYOLModel
from models.moco    import MoCoModel
from models.simsiam import SimSiamModel
from models.dino    import DINOModel

class JointModel(pl.LightningModule):
    def __init__(self,
                 selected_ssl: str,
                 ssl_params: dict,
                 cinn_params: dict,
                 λ_contrast: float = 1.0,
                 λ_cinn:     float = 1.0):
        super().__init__()
        # Turn off auto‐opt so we can use two optimizers
        self.automatic_optimization = False

        self.save_hyperparameters()

        # 1) Instantiate your SSL module
        if selected_ssl == "simclr":
            ssl_mod = SimCLRModel(**ssl_params)
        elif selected_ssl == "byol":
            ssl_mod = BYOLModel(**ssl_params)
        elif selected_ssl == "moco":
            ssl_mod = MoCoModel(**ssl_params)
        elif selected_ssl == "simsiam":
            ssl_mod = SimSiamModel(**ssl_params)
        elif selected_ssl == "dino":
            ssl_mod = DINOModel(**ssl_params)
        else:
            raise ValueError(f"Unknown SSL model '{selected_ssl}'")

        # Pull out backbone, heads, loss
        self.backbone        = ssl_mod.backbone
        self.projection_head = getattr(ssl_mod, "projection_head", None)
        self.prediction_head = getattr(ssl_mod, "prediction_head", None)
        self.ssl_crit        = ssl_mod.criterion

        # 2) Build the cINN
        self.cinn = cINN(
            y_dim      = cinn_params["y_dim"],
            x_dim      = cinn_params["x_dim"],
            hidden_dim = cinn_params["hidden_dim"],
            n_blocks   = cinn_params["n_blocks"],
            clamp      = cinn_params["clamp"],
        )

        # 3) Loss weights & learning rates
        self.λ_contrast = λ_contrast
        self.λ_cinn     = λ_cinn
        self.lr_ssl     = ssl_params["learning_rate"]
        self.lr_cinn    = cinn_params["learning_rate"]

    def training_step(self, batch, batch_idx):
        # grab the two optimizers
        opt_ssl, opt_cinn = self.optimizers()

        (x0, x1), y = batch
        if batch_idx == 0:
           print(f"[DBG] batch_idx=0 x0 nan={torch.isnan(x0).sum()}, "
                 f"x1 nan={torch.isnan(x1).sum()}, y nan={torch.isnan(y).sum()}")
           print(f"[DBG] y min/max: {y.min().item():.3f}/{y.max().item():.3f}")
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        y  = y.to(self.device)

        # 1) contrastive forward + loss
        h0 = self.backbone(x0).flatten(1)
        if batch_idx == 0:
            print(f"[DBG] h0 mean/std/nan: {h0.mean().item():.3f}/"
                 f"{h0.std().item():.3f}/{int(torch.isnan(h0).sum())}")
        h1 = self.backbone(x1).flatten(1)
        z0 = self.projection_head(h0)
        z1 = self.projection_head(h1)

        # --- normalize embeddings to unit ℓ² (SimCLR style) ---
        z0n = torch.nn.functional.normalize(z0, dim=1)
        z1n = torch.nn.functional.normalize(z1, dim=1)
        # compute InfoNCE on normalized embeddings
        Lc = self.ssl_crit(z0n, z1n)

        # debug if Lc is NaN
        if torch.isnan(Lc):
            print(f"[DBG] batch {batch_idx}: InfoNCE loss is NaN")
            print(f"[DBG] z0n mean/std: {z0n.mean().item():.4f}/{z0n.std().item():.4f}")
            print(f"[DBG] z1n mean/std: {z1n.mean().item():.4f}/{z1n.std().item():.4f}")
            # show first few entries
            print("  z0n[0,:5]:", z0n[0,:5].tolist())
            print("  z1n[0,:5]:", z1n[0,:5].tolist())
            raise RuntimeError("NaN in InfoNCE loss")


        # 2) cINN forward + NLL loss
        z_cinn, log_jac = self.cinn(y, h0.detach())
        if batch_idx == 0:
            print(f"[DBG] z_cinn mean/std/nan: {z_cinn.mean().item():.3f}/"
                 f"{z_cinn.std().item():.3f}/{int(torch.isnan(z_cinn).sum())}")
            print(f"[DBG] log_jac mean/std/nan: {log_jac.mean().item():.3f}/"
                 f"{log_jac.std().item():.3f}/{int(torch.isnan(log_jac).sum())}")
        Ln = 0.5*(z_cinn**2).sum(1).mean() - log_jac.mean()

        # 3) total loss
        L = self.λ_contrast * Lc + self.λ_cinn * Ln

        # Manually backprop & step both optimizers
        self.manual_backward(L)

        opt_ssl.step()
        opt_cinn.step()

        opt_ssl.zero_grad(set_to_none=True)
        opt_cinn.zero_grad(set_to_none=True)

        # Logging
        self.log("loss_contrast", Lc, on_step=True, prog_bar=True)
        self.log("loss_cinn",     Ln, on_step=True, prog_bar=True)
        self.log("loss_total",    L,  on_step=True, prog_bar=True)

        return L

    def configure_optimizers(self):
        # SSL optimizer (SGD)
        ssl_params = list(self.backbone.parameters())
        ssl_params += list(self.projection_head.parameters())
        opt_ssl = torch.optim.SGD(
            ssl_params,
            lr=self.lr_ssl,
            momentum=0.9,
            weight_decay=5e-4
        )
        # cINN optimizer (Adam)
        opt_cinn = torch.optim.Adam(
            self.cinn.parameters(),
            lr=self.lr_cinn,
            weight_decay=1e-6
        )
        return [opt_ssl, opt_cinn]
