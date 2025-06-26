import os
from datetime import datetime
import warnings

import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from metrics import cox_log_rank, CIndex_lifeline, MultiLabel_Acc
from losses import surv_loss, MultiTaskLossWrapper
from models import AutoEncoder_New
from data_utils import get_transforms, get_dataloaders
from monai.data import Dataset, DataLoader, list_data_collate

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["train_validate_test_ae"]

# ------------------------------------------------------------------
# Helper: evaluation & inference for AE model
# ------------------------------------------------------------------

def eval_ae(model, files, transforms, batch_size=40):

    loader = DataLoader(
        Dataset(files, transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    with torch.no_grad():
        os_evt = os_time = pfs_evt = pfs_time = None
        os_out = pfs_out = age_out = bottleneck = ids = None

        for i, batch in enumerate(loader):
            x = batch["input"].cuda()
            o, p, a, _, _, feat = model(x)

            if i == 0:
                os_evt = batch["OS_status"].cuda()
                os_time = batch["OS_time"].cuda()
                pfs_evt = batch["PFS_status"].cuda()
                pfs_time = batch["PFS_time"].cuda()

                os_out = o
                pfs_out = p
                age_out = a
                bottleneck = feat
                ids = torch.tensor([int(k) for k in batch["ID"]]).cuda()
            else:
                os_evt = torch.cat([os_evt, batch["OS_status"].cuda()])
                os_time = torch.cat([os_time, batch["OS_time"].cuda()])
                pfs_evt = torch.cat([pfs_evt, batch["PFS_status"].cuda()])
                pfs_time = torch.cat([pfs_time, batch["PFS_time"].cuda()])

                os_out = torch.cat([os_out, o])
                pfs_out = torch.cat([pfs_out, p])
                age_out = torch.cat([age_out, a])
                bottleneck = torch.cat([bottleneck, feat])
                ids = torch.cat(
                    [ids, torch.tensor([int(k) for k in batch["ID"]]).cuda()]
                )

        print(
            f"[Eval] OS c-index {CIndex_lifeline(os_out, os_evt, os_time):.4f}, "
            f"PFS c-index {CIndex_lifeline(pfs_out, pfs_evt, pfs_time):.4f}"
        )
        return (
            os_out,
            pfs_out,
            os_evt,
            os_time,
            pfs_evt,
            pfs_time,
            ids,
            age_out,
            bottleneck,
        )

# ------------------------------------------------------------------
# Training / validation / test loop
# ------------------------------------------------------------------

def train_validate_test_ae(
    train_files,
    val_files,
    test_files,
    cfg,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------
    train_tf, val_tf = get_transforms(cfg["rand_p"])
    loaders = get_dataloaders(
        train_files,
        val_files,
        test_files,
        train_tf,
        val_tf,
        batch_sizes=(cfg["train_bs"], cfg["val_bs"], cfg["test_bs"]),
    )
    train_loader, val_loader, test_loader = loaders

    # --------------------------------------------------------------
    # Model & losses
    # --------------------------------------------------------------
    model = AutoEncoder_New(
        dimensions=3,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        inter_channels=[256, 256],
        inter_dilations=[2, 2],
        dropout=cfg["drop_rate"],
    ).to(device)

    multitask_loss = MultiTaskLossWrapper(task_num=5).to(device)
    loss_mse = torch.nn.MSELoss()
    loss_l1 = torch.nn.L1Loss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=cfg["lr_decay"]
    )

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------
    example_batch = next(iter(train_loader))
    summary(model, input_size=example_batch["input"].shape)

    tb_dir = (
        "runs/"
        + datetime.now().strftime("%Y-%m-%d-T%H-%M-%S/")
        + cfg["name"]
    )
    writer = SummaryWriter(log_dir=tb_dir)

    best_os = best_pfs = -1.0

    bar = trange(cfg["epochs"], desc="epoch 0", leave=True)
    for epoch in bar:
        # ============================ TRAIN ============================ #
        model.train()
        epoch_loss = 0.0

        # Containers for metrics
        tr_os_evt = tr_os_time = tr_pfs_evt = tr_pfs_time = None
        tr_os_out = tr_pfs_out = tr_label = tr_label_out = None

        for step, batch in enumerate(train_loader, 1):
            x = batch["input"].to(device)

            # Survival targets
            os_evt = batch["OS_status"].to(device)
            os_time = batch["OS_time"].to(device)
            pfs_evt = batch["PFS_status"].to(device)
            pfs_time = batch["PFS_time"].to(device)

            # Auxiliary targets
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

            # Forward
            os_out, pfs_out, age_out, cls_out, recon, _ = model(x)

            # Survival loss
            loss_os = surv_loss(os_evt, os_time, os_out)
            loss_pfs = surv_loss(pfs_evt, pfs_time, pfs_out)

            # Auxiliary
            loss_age = loss_mse(
                age_out, age.unsqueeze(1).float().log_() / 4.75
            )
            loss_cls = loss_bce(cls_out, cls_target.float())
            loss_img = loss_l1(recon, x)

            # Total
            total_loss = multitask_loss(
                loss_pfs, loss_age, loss_cls, loss_img, loss_os
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # concat for epoch metrics
            if step == 1:
                tr_os_evt, tr_os_time = os_evt, os_time
                tr_pfs_evt, tr_pfs_time = pfs_evt, pfs_time
                tr_os_out, tr_pfs_out = os_out, pfs_out
                tr_label, tr_label_out = cls_target, cls_out
            else:
                tr_os_evt = torch.cat([tr_os_evt, os_evt])
                tr_os_time = torch.cat([tr_os_time, os_time])
                tr_pfs_evt = torch.cat([tr_pfs_evt, pfs_evt])
                tr_pfs_time = torch.cat([tr_pfs_time, pfs_time])
                tr_os_out = torch.cat([tr_os_out, os_out])
                tr_pfs_out = torch.cat([tr_pfs_out, pfs_out])
                tr_label = torch.cat([tr_label, cls_target], 0)
                tr_label_out = torch.cat([tr_label_out, cls_out], 0)

            # step logging
            gs = epoch * len(train_loader) + step
            writer.add_scalar("train/total_loss_step", total_loss.item(), gs)

        epoch_loss /= len(train_loader)
        bar.set_description(f"epoch {epoch+1}, avg loss {epoch_loss:.4f}")
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # epoch metrics
        t_c_os = CIndex_lifeline(tr_os_out, tr_os_evt, tr_os_time)
        t_c_pfs = CIndex_lifeline(tr_pfs_out, tr_pfs_evt, tr_pfs_time)
        t_lbl_pred = (tr_label_out >= 0)
        t_acc = MultiLabel_Acc(t_lbl_pred, tr_label)
        writer.add_scalar("train/cindex_OS", t_c_os, epoch)
        writer.add_scalar("train/cindex_PFS", t_c_pfs, epoch)
        for i, a in enumerate(t_acc):
            writer.add_scalar(f"train/acc_label_{i}", a, epoch)

        # ============================ VALIDATION ============================ #
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

                os_out, pfs_out, age_out, cls_out, recon, _ = model(x)

                v_loss = multitask_loss(
                    surv_loss(pfs_evt, pfs_time, pfs_out),
                    loss_mse(age_out, age.unsqueeze(1).float().log_() / 4.75),
                    loss_bce(cls_out, cls_target.float()),
                    loss_l1(recon, x),
                    surv_loss(os_evt, os_time, os_out),
                )
                val_loss += v_loss.item()

            val_loss /= len(val_loader)
            writer.add_scalar("val/epoch_loss", val_loss, epoch)
            scheduler.step(val_loss)

        # ============================ TEST & SAVE ============================ #
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_loader:
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

                os_out, pfs_out, age_out, cls_out, recon, _ = model(x)
                te_loss = multitask_loss(
                    surv_loss(pfs_evt, pfs_time, pfs_out),
                    loss_mse(age_out, age.unsqueeze(1).float().log_() / 4.75),
                    loss_bce(cls_out, cls_target.float()),
                    loss_l1(recon, x),
                    surv_loss(os_evt, os_time, os_out),
                )
                test_loss += te_loss.item()
            test_loss /= len(test_loader)
            writer.add_scalar("test/epoch_loss", test_loss, epoch)

        # save best c-index models
        if epoch > cfg["skip_epoch_model"]:
            te_c_os = CIndex_lifeline(tr_os_out, tr_os_evt, tr_os_time)
            te_c_pfs = CIndex_lifeline(tr_pfs_out, tr_pfs_evt, tr_pfs_time)

            if te_c_pfs > best_pfs:
                best_pfs = te_c_pfs
                torch.save(model.state_dict(), f"{cfg['name']}_PFS.pth")
            if te_c_os > best_os:
                best_os = te_c_os
                torch.save(model.state_dict(), f"{cfg['name']}_OS.pth")

    writer.close()
    return model
