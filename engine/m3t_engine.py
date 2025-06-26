import os
from datetime import datetime
import warnings

import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from models import M3T
from metrics import cox_log_rank, CIndex_lifeline, MultiLabel_Acc
from losses import surv_loss, MultiTaskLossWrapper
from data_utils import get_transforms, get_dataloaders

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["train_validate_test_m3t"]


def train_validate_test_m3t(
    train_files, val_files, test_files, cfg, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_tf, val_tf = get_transforms(cfg["rand_p"])
    loaders = get_dataloaders(
        train_files, val_files, test_files,
        train_tf, val_tf,
        batch_sizes=(cfg["train_bs"], cfg["val_bs"], cfg["test_bs"])
    )
    train_loader, val_loader, test_loader = loaders

    # ------------------------------------------------------------------
    # Model & losses
    # ------------------------------------------------------------------
    model = M3T(
        in_channels=3,
        out_channels=32,
        proj_dim=768,
        depth=cfg["depth"],
        heads=8,
        drop_p=cfg["drop_rate"],
    ).to(device)

    multitask_loss = MultiTaskLossWrapper(task_num=4).to(device)
    loss_mse = torch.nn.MSELoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.Adam(model.parameters(), cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, "min", factor=cfg["lr_decay"]
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    example = next(iter(train_loader))
    summary(model, input_size=example["input"].shape)
    tb_dir = (
        "runs/"
        + datetime.now().strftime("%Y-%m-%d-T%H-%M-%S/")
        + cfg["name"]
    )
    writer = SummaryWriter(log_dir=tb_dir)

    best_os = best_pfs = -1.0
    bar = trange(cfg["epochs"], desc="epoch 0", leave=True)

    for epoch in bar:
        model.train()
        ep_loss = 0.0

        # Containers for metrics
        tr_os_evt = tr_os_time = tr_pfs_evt = tr_pfs_time = None
        tr_os_out = tr_pfs_out = tr_lbl = tr_lbl_out = None

        for step, batch in enumerate(train_loader, 1):
            x = batch["input"].to(device)
            os_evt = batch["OS_status"].to(device)
            os_time = batch["OS_time"].to(device)
            pfs_evt = batch["PFS_status"].to(device)
            pfs_time = batch["PFS_time"].to(device)

            age = batch["Age"].to(device)
            gender = batch["Gender"].to(device).unsqueeze(1)
            cp = batch["Child-Pugh"].to(device).unsqueeze(1)
            hbv = batch["HBV_infection"].to(device).unsqueeze(1)
            pvtt = batch["PVTT"].to(device).unsqueeze(1)
            lmet = batch["LungMet"].to(device).unsqueeze(1)
            bmet = batch["BoneMet"].to(device).unsqueeze(1)
            seven = batch["Up_to_seven"].to(device).unsqueeze(1)
            lnmet = batch["LNMet"].to(device).unsqueeze(1)
            stage = batch["Stage"].to(device).unsqueeze(1)
            cls_target = torch.cat(
                (gender, cp, hbv, pvtt, lmet, bmet, seven, lnmet, stage), 1
            )

            os_out, pfs_out, age_out, cls_out = model(x)

            loss_os = surv_loss(os_evt, os_time, os_out)
            loss_pfs = surv_loss(pfs_evt, pfs_time, pfs_out)
            loss_age = loss_mse(
                age_out, age.unsqueeze(1).float().log_() / 4.75
            )
            loss_cls = loss_bce(cls_out, cls_target.float())
            total = multitask_loss(loss_os, loss_pfs, loss_age, loss_cls)

            optim.zero_grad()
            total.backward()
            optim.step()

            ep_loss += total.item()

            # metrics container
            if step == 1:
                tr_os_evt, tr_os_time = os_evt, os_time
                tr_pfs_evt, tr_pfs_time = pfs_evt, pfs_time
                tr_os_out, tr_pfs_out = os_out, pfs_out
                tr_lbl, tr_lbl_out = cls_target, cls_out
            else:
                tr_os_evt = torch.cat([tr_os_evt, os_evt])
                tr_os_time = torch.cat([tr_os_time, os_time])
                tr_pfs_evt = torch.cat([tr_pfs_evt, pfs_evt])
                tr_pfs_time = torch.cat([tr_pfs_time, pfs_time])
                tr_os_out = torch.cat([tr_os_out, os_out])
                tr_pfs_out = torch.cat([tr_pfs_out, pfs_out])
                tr_lbl = torch.cat([tr_lbl, cls_target], 0)
                tr_lbl_out = torch.cat([tr_lbl_out, cls_out], 0)

        ep_loss /= len(train_loader)
        bar.set_description(f"epoch {epoch+1}, loss {ep_loss:.4f}")
        writer.add_scalar("train/epoch_loss", ep_loss, epoch)

        # epoch metrics
        c_os = CIndex_lifeline(tr_os_out, tr_os_evt, tr_os_time)
        c_pfs = CIndex_lifeline(tr_pfs_out, tr_pfs_evt, tr_pfs_time)
        lbl_pred = (tr_lbl_out >= 0)
        acc = MultiLabel_Acc(lbl_pred, tr_lbl)
        writer.add_scalar("train/cindex_OS", c_os, epoch)
        writer.add_scalar("train/cindex_PFS", c_pfs, epoch)
        for i, a in enumerate(acc):
            writer.add_scalar(f"train/acc_label_{i}", a, epoch)

        # ---------------- validation ----------------
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                x = batch["input"].to(device)
                os_evt = batch["OS_status"].to(device)
                os_time = batch["OS_time"].to(device)
                pfs_evt = batch["PFS_status"].to(device)
                pfs_time = batch["PFS_time"].to(device)

                age = batch["Age"].to(device)
                gender = batch["Gender"].to(device).unsqueeze(1)
                cp = batch["Child-Pugh"].to(device).unsqueeze(1)
                hbv = batch["HBV_infection"].to(device).unsqueeze(1)
                pvtt = batch["PVTT"].to(device).unsqueeze(1)
                lmet = batch["LungMet"].to(device).unsqueeze(1)
                bmet = batch["BoneMet"].to(device).unsqueeze(1)
                seven = batch["Up_to_seven"].to(device).unsqueeze(1)
                lnmet = batch["LNMet"].to(device).unsqueeze(1)
                stage = batch["Stage"].to(device).unsqueeze(1)
                cls_target = torch.cat(
                    (gender, cp, hbv, pvtt, lmet, bmet, seven, lnmet, stage),
                    1,
                )

                os_out, pfs_out, age_out, cls_out = model(x)
                v_loss = multitask_loss(
                    surv_loss(os_evt, os_time, os_out),
                    surv_loss(pfs_evt, pfs_time, pfs_out),
                    loss_mse(age_out, age.unsqueeze(1).float().log_() / 4.75),
                    loss_bce(cls_out, cls_target.float()),
                )
                val_loss += v_loss.item()

            val_loss /= len(val_loader)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)
            sched.step(val_loss)

        # --------------- test & save checkpoints ---------------
        if epoch > cfg["skip_epoch_model"]:
            if c_pfs > best_pfs:
                best_pfs = c_pfs
                torch.save(model.state_dict(), f"{cfg['name']}_PFS.pth")
            if c_os > best_os:
                best_os = c_os
                torch.save(model.state_dict(), f"{cfg['name']}_OS.pth")

    writer.close()
    return model
