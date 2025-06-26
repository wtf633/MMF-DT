import os
from datetime import datetime
import warnings

import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from metrics import (
    cox_log_rank,
    CIndex_lifeline,
    MultiLabel_Acc,
)
from losses import (
    surv_loss,
    MultiTaskLossWrapper,
)
from models import EfficientNet

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["train_validate_test"]


def train_validate_test(train_loader,
                        val_loader,
                        test_loader,
                        device,
                        cfg):
    """
    Full training / validation / test pipeline.
    """
    # Hyper-parameters from cfg
    lr = cfg.get("lr", 1e-5)
    lr_decay = cfg.get("lr_decay", 0.05)
    max_epochs = cfg.get("epochs", 250)
    rand_p = cfg.get("rand_p", 0.55)
    n_loss = cfg.get("n_loss", 4)
    skip_epoch_model = cfg.get("skip_epoch_model", 50)
    best_name = cfg.get("name", "multitask_subnet1")

    # Model, losses, optimizer
    model = EfficientNet().to(device)
    loss_wrapper = MultiTaskLossWrapper(n_loss).to(device)
    loss_mse = torch.nn.MSELoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=lr_decay
    )

    # Print model summary
    example_batch = next(iter(train_loader))
    summary(model, input_size=example_batch["input"].shape)

    # TensorBoard
    run_dir = ("runs/" +
               datetime.now().strftime("%Y-%m-%d-T%H-%M-%S/") +
               best_name)
    writer = SummaryWriter(log_dir=run_dir)

    best_metric_os = -1.0
    best_metric_pfs = -1.0
    best_epoch = -1

    epoch_bar = trange(max_epochs, desc="epoch 0, avg loss: inf", leave=True)

    for epoch in epoch_bar:
        # --------------------------- TRAIN --------------------------- #
        model.train()
        epoch_loss = 0.0
        step = 0

        # Containers for whole-epoch metrics
        t_os_evt = t_os_time = t_pfs_evt = t_pfs_time = None
        t_os_out = t_pfs_out = None
        t_label = t_label_out = None

        for batch in train_loader:
            inputs = batch["input"].to(device)

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

            label_cat = torch.cat(
                (gender, cp, hbv, pvtt, lmet, bmet, seven, lnmet, stage), 1
            )

            # Forward
            os_out, pfs_out, age_out, lbl_out = model(inputs)

            # Concatenate for epoch-level stats
            if step == 0:
                t_os_evt, t_os_time, t_os_out = os_evt, os_time, os_out
                t_pfs_evt, t_pfs_time, t_pfs_out = pfs_evt, pfs_time, pfs_out
                t_label, t_label_out = label_cat, lbl_out
            else:
                t_os_evt = torch.cat([t_os_evt, os_evt])
                t_os_time = torch.cat([t_os_time, os_time])
                t_os_out = torch.cat([t_os_out, os_out])

                t_pfs_evt = torch.cat([t_pfs_evt, pfs_evt])
                t_pfs_time = torch.cat([t_pfs_time, pfs_time])
                t_pfs_out = torch.cat([t_pfs_out, pfs_out])

                t_label = torch.cat([t_label, label_cat], 0)
                t_label_out = torch.cat([t_label_out, lbl_out], 0)

            # Losses
            loss_os = surv_loss(os_evt, os_time, os_out)
            loss_pfs = surv_loss(pfs_evt, pfs_time, pfs_out)
            loss_age = loss_mse(
                age_out, age.unsqueeze(1).float().log_() / 4.75
            )
            loss_lbl = loss_bce(lbl_out, label_cat.float())

            # L1 regularization
            l1_reg = None
            for w in model.parameters():
                l1 = torch.abs(w).sum()
                l1_reg = l1 if l1_reg is None else l1_reg + l1

            total_loss = (
                loss_wrapper(loss_os, loss_pfs, loss_age, loss_lbl)
                * torch.log10(l1_reg)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += total_loss.item()

            # Per-step TensorBoard logs
            epoch_size = len(train_loader)
            global_step = epoch * epoch_size + step
            writer.add_scalar("train/total_loss_step", total_loss.item(),
                              global_step)
            writer.add_scalar("train/os_loss_step", loss_os.item(),
                              global_step)
            writer.add_scalar("train/pfs_loss_step", loss_pfs.item(),
                              global_step)
            writer.add_scalar("train/age_loss_step", loss_age.item(),
                              global_step)
            writer.add_scalar("train/label_loss_step", loss_lbl.item(),
                              global_step)
            writer.add_scalar("train/l1_loss_step", l1_reg.item(),
                              global_step)

            torch.cuda.empty_cache()

        epoch_loss /= step
        epoch_bar.set_description(
            f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}"
        )
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # Epoch-level metrics
        p_os = cox_log_rank(t_os_out, t_os_evt, t_os_time)
        c_os = CIndex_lifeline(t_os_out, t_os_evt, t_os_time)
        p_pfs = cox_log_rank(t_pfs_out, t_pfs_evt, t_pfs_time)
        c_pfs = CIndex_lifeline(t_pfs_out, t_pfs_evt, t_pfs_time)
        lbl_pred = (t_label_out >= 0)
        acc = MultiLabel_Acc(lbl_pred, t_label)

        writer.add_scalar("train/logrank_OS", p_os, epoch)
        writer.add_scalar("train/cindex_OS", c_os, epoch)
        writer.add_scalar("train/logrank_PFS", p_pfs, epoch)
        writer.add_scalar("train/cindex_PFS", c_pfs, epoch)
        for i, a in enumerate(acc):
            writer.add_scalar(f"train/acc_label_{i}", a, epoch)

        # ------------------------- VALIDATION ------------------------ #
        model.eval()
        val_loss_epoch = 0.0
        v_step = 0

        v_os_evt = v_os_time = v_pfs_evt = v_pfs_time = None
        v_os_out = v_pfs_out = None
        v_label = v_label_out = None

        with torch.no_grad():
            for batch in val_loader:
                v_in = batch["input"].to(device)

                v_os_evt_b = batch["OS_status"].to(device)
                v_os_time_b = batch["OS_time"].to(device)
                v_pfs_evt_b = batch["PFS_status"].to(device)
                v_pfs_time_b = batch["PFS_time"].to(device)

                v_age = batch["Age"].to(device)
                v_gender = batch["Gender"].to(device).unsqueeze(1)
                v_cp = batch["Child-Pugh"].to(device).unsqueeze(1)
                v_hbv = batch["HBV_infection"].to(device).unsqueeze(1)
                v_pvtt = batch["PVTT"].to(device).unsqueeze(1)
                v_lmet = batch["LungMet"].to(device).unsqueeze(1)
                v_bmet = batch["BoneMet"].to(device).unsqueeze(1)
                v_seven = batch["Up_to_seven"].to(device).unsqueeze(1)
                v_lnmet = batch["LNMet"].to(device).unsqueeze(1)
                v_stage = batch["Stage"].to(device).unsqueeze(1)

                v_lbl_cat = torch.cat(
                    (v_gender, v_cp, v_hbv, v_pvtt, v_lmet,
                     v_bmet, v_seven, v_lnmet, v_stage), 1
                )

                v_os_out_b, v_pfs_out_b, v_age_out_b, v_lbl_out_b = model(v_in)

                # Concatenate for epoch metrics
                if v_step == 0:
                    v_os_evt, v_os_time = v_os_evt_b, v_os_time_b
                    v_pfs_evt, v_pfs_time = v_pfs_evt_b, v_pfs_time_b
                    v_os_out, v_pfs_out = v_os_out_b, v_pfs_out_b
                    v_label, v_label_out = v_lbl_cat, v_lbl_out_b
                else:
                    v_os_evt = torch.cat([v_os_evt, v_os_evt_b])
                    v_os_time = torch.cat([v_os_time, v_os_time_b])
                    v_pfs_evt = torch.cat([v_pfs_evt, v_pfs_evt_b])
                    v_pfs_time = torch.cat([v_pfs_time, v_pfs_time_b])
                    v_os_out = torch.cat([v_os_out, v_os_out_b])
                    v_pfs_out = torch.cat([v_pfs_out, v_pfs_out_b])
                    v_label = torch.cat([v_label, v_lbl_cat], 0)
                    v_label_out = torch.cat([v_label_out, v_lbl_out_b], 0)

                v_loss_os = surv_loss(v_os_evt_b, v_os_time_b, v_os_out_b)
                v_loss_pfs = surv_loss(v_pfs_evt_b, v_pfs_time_b, v_pfs_out_b)
                v_loss_age = loss_mse(
                    v_age_out_b, v_age.unsqueeze(1).float().log_() / 4.75
                )
                v_loss_lbl = loss_bce(v_lbl_out_b, v_lbl_cat.float())

                # The original script multiplies by v_loss_os
                v_total_loss = (
                    loss_wrapper(v_loss_os, v_loss_pfs,
                                 v_loss_age, v_loss_lbl)
                )
                v_step += 1
                val_loss_epoch += v_total_loss.item()

                idx = len(val_loader)
                writer.add_scalar("val/total_loss_step",
                                  v_total_loss.item(),
                                  epoch * idx + v_step)

        val_loss_epoch /= v_step
        scheduler.step(val_loss_epoch)
        writer.add_scalar("val/epoch_loss", val_loss_epoch, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch + 1)

        # Validation metrics
        v_p_os = cox_log_rank(v_os_out, v_os_evt, v_os_time)
        v_c_os = CIndex_lifeline(v_os_out, v_os_evt, v_os_time)
        v_p_pfs = cox_log_rank(v_pfs_out, v_pfs_evt, v_pfs_time)
        v_c_pfs = CIndex_lifeline(v_pfs_out, v_pfs_evt, v_pfs_time)
        v_lbl_pred = (v_label_out >= 0)
        v_acc = MultiLabel_Acc(v_lbl_pred, v_label)

        writer.add_scalar("val/logrank_OS", v_p_os, epoch)
        writer.add_scalar("val/cindex_OS", v_c_os, epoch)
        writer.add_scalar("val/logrank_PFS", v_p_pfs, epoch)
        writer.add_scalar("val/cindex_PFS", v_c_pfs, epoch)
        for i, a in enumerate(v_acc):
            writer.add_scalar(f"val/acc_label_{i}", a, epoch)

        # --------------------------- TEST --------------------------- #
        model.eval()
        test_loss_epoch = 0.0
        s_step = 0

        s_os_evt = s_os_time = s_pfs_evt = s_pfs_time = None
        s_os_out = s_pfs_out = None
        s_label = s_label_out = None

        with torch.no_grad():
            for batch in test_loader:
                s_in = batch["input"].to(device)

                s_os_evt_b = batch["OS_status"].to(device)
                s_os_time_b = batch["OS_time"].to(device)
                s_pfs_evt_b = batch["PFS_status"].to(device)
                s_pfs_time_b = batch["PFS_time"].to(device)

                s_age = batch["Age"].to(device)
                s_gender = batch["Gender"].to(device).unsqueeze(1)
                s_cp = batch["Child-Pugh"].to(device).unsqueeze(1)
                s_hbv = batch["HBV_infection"].to(device).unsqueeze(1)
                s_pvtt = batch["PVTT"].to(device).unsqueeze(1)
                s_lmet = batch["LungMet"].to(device).unsqueeze(1)
                s_bmet = batch["BoneMet"].to(device).unsqueeze(1)
                s_seven = batch["Up_to_seven"].to(device).unsqueeze(1)
                s_lnmet = batch["LNMet"].to(device).unsqueeze(1)
                s_stage = batch["Stage"].to(device).unsqueeze(1)

                s_lbl_cat = torch.cat(
                    (s_gender, s_cp, s_hbv, s_pvtt, s_lmet,
                     s_bmet, s_seven, s_lnmet, s_stage), 1
                )

                s_os_out_b, s_pfs_out_b, s_age_out_b, s_lbl_out_b = model(s_in)

                if s_step == 0:
                    s_os_evt, s_os_time = s_os_evt_b, s_os_time_b
                    s_pfs_evt, s_pfs_time = s_pfs_evt_b, s_pfs_time_b
                    s_os_out, s_pfs_out = s_os_out_b, s_pfs_out_b
                    s_label, s_label_out = s_lbl_cat, s_lbl_out_b
                else:
                    s_os_evt = torch.cat([s_os_evt, s_os_evt_b])
                    s_os_time = torch.cat([s_os_time, s_os_time_b])
                    s_pfs_evt = torch.cat([s_pfs_evt, s_pfs_evt_b])
                    s_pfs_time = torch.cat([s_pfs_time, s_pfs_time_b])
                    s_os_out = torch.cat([s_os_out, s_os_out_b])
                    s_pfs_out = torch.cat([s_pfs_out, s_pfs_out_b])
                    s_label = torch.cat([s_label, s_lbl_cat], 0)
                    s_label_out = torch.cat([s_label_out, s_lbl_out_b], 0)

                s_loss_os = surv_loss(s_os_evt_b, s_os_time_b, s_os_out_b)
                s_loss_pfs = surv_loss(
                    s_pfs_evt_b, s_pfs_time_b, s_pfs_out_b
                )
                s_loss_age = loss_mse(
                    s_age_out_b, s_age.unsqueeze(1).float().log_() / 4.75
                )
                s_loss_lbl = loss_bce(s_lbl_out_b, s_lbl_cat.float())

                s_total_loss = loss_wrapper(
                    s_loss_os, s_loss_pfs, s_loss_age, s_loss_lbl
                )
                s_step += 1
                test_loss_epoch += s_total_loss.item()

                idx_test = len(test_loader)
                writer.add_scalar("test/total_loss_step",
                                  s_total_loss.item(),
                                  epoch * idx_test + s_step)

        test_loss_epoch /= s_step
        writer.add_scalar("test/epoch_loss", test_loss_epoch, epoch)

        s_p_os = cox_log_rank(s_os_out, s_os_evt, s_os_time)
        s_c_os = CIndex_lifeline(s_os_out, s_os_evt, s_os_time)
        s_p_pfs = cox_log_rank(s_pfs_out, s_pfs_evt, s_pfs_time)
        s_c_pfs = CIndex_lifeline(s_pfs_out, s_pfs_evt, s_pfs_time)
        s_lbl_pred = (s_label_out >= 0)
        s_acc = MultiLabel_Acc(s_lbl_pred, s_label)

        writer.add_scalar("test/logrank_OS", s_p_os, epoch)
        writer.add_scalar("test/cindex_OS", s_c_os, epoch)
        writer.add_scalar("test/logrank_PFS", s_p_pfs, epoch)
        writer.add_scalar("test/cindex_PFS", s_c_pfs, epoch)
        for i, a in enumerate(s_acc):
            writer.add_scalar(f"test/acc_label_{i}", a, epoch)

        # ---------------------- SAVE BEST MODELS -------------------- #
        if epoch > skip_epoch_model:
            if s_c_pfs > best_metric_pfs:
                best_metric_pfs = s_c_pfs
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f"{best_name}_PFS.pth")
            if s_c_os > best_metric_os:
                best_metric_os = s_c_os
                torch.save(model.state_dict(), f"{best_name}_OS.pth")

    writer.close()
    return model
