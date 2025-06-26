import os
import torch
import pandas as pd
from data_utils import (
    build_file_dicts,
    get_transforms,
    get_dataloaders,
)
from engine import train_validate_test
from inference import model_eval_gpu, model_run_gpu
import argparse

pd.options.mode.chained_assignment = None  # silence pandas warning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ----------------- Configuration ----------------- #
    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument('--name', default='multitask_subnet1')
        p.add_argument('--lr', type=float, default=1e-5)
        p.add_argument('--lr_decay', type=float, default=0.05)
        p.add_argument('--epochs', type=int, default=250)
        p.add_argument('--rand_p', type=float, default=0.55)
        p.add_argument('--train_bs', type=int, default=16)
        p.add_argument('--val_bs', type=int, default=8)
        p.add_argument('--test_bs', type=int, default=8)
        p.add_argument('--skip_epoch_model', type=int, default=50)
        # paths
        p.add_argument('--train_img_dir', default='path/to/train_imgs')
        p.add_argument('--val_img_dir', default='path/to/val_imgs')
        p.add_argument('--test_img_dir', default='path/to/test_imgs')
        p.add_argument('--csv_root', default='path/to/csv_root')
        p.add_argument('--gpu', default='0')
        return p.parse_args()

    # ==== inside main() ====
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = vars(args)

    # ----------------- Data Dictionaries ----------------- #
    train_files = build_file_dicts(
        cfg['train_img_dir'], f"{cfg['csv_root']}/train_events.csv"
    )
    val_files = build_file_dicts(
        cfg['val_img_dir'], f"{cfg['csv_root']}/valid_events.csv"
    )
    test_files = build_file_dicts(
        cfg['test_img_dir'], f"{cfg['csv_root']}/test_events.csv"
    )

    # ----------------- Transforms & Loaders ----------------- #
    train_tf, val_tf = get_transforms(cfg["rand_p"])
    loaders = get_dataloaders(
        train_files, val_files, test_files,
        train_tf, val_tf,
        batch_sizes=(16, 8, 8)
    )
    train_loader, val_loader, test_loader = loaders

    # ----------------- Training ----------------- #
    model = train_validate_test(
        train_loader, val_loader, test_loader,
        device, cfg
    )

    # ----------------- Best Model Evaluation ----------------- #
    model.load_state_dict(torch.load(f"{cfg['name']}_PFS.pth"))
    _, pfs_risk_train, os_train, ostime_train, pfs_train, pfstime_train, ID_train, age_train = model_eval_gpu(model,
                                                                                                              train_files,
                                                                                                              val_tf)
    _, pfs_risk_val, os_val, ostime_val, pfs_val, pfstime_val, ID_val, age_val = model_eval_gpu(model, val_files,
                                                                                                val_tf)
    _, pfs_risk_test, os_test, ostime_test, pfs_test, pfstime_test, ID_test, age_test = model_eval_gpu(model,
                                                                                                       test_files,
                                                                                                       val_tf)

    model.load_state_dict(torch.load(f"{cfg['name']}_OS.pth"))
    os_risk_train, _, _, _, _, _, _, _ = model_eval_gpu(model, train_files, val_tf)
    os_risk_val, _, _, _, _, _, _, _ = model_eval_gpu(model, val_files, val_tf)
    os_risk_test, _, _, _, _, _, _, _ = model_eval_gpu(model, test_files, val_tf)

    os_train, ostime_train, pfs_train, pfstime_train, ID_train = os_train.unsqueeze(1), ostime_train.unsqueeze(
        1), pfs_train.unsqueeze(1), pfstime_train.unsqueeze(1), ID_train.unsqueeze(1)
    os_val, ostime_val, pfs_val, pfstime_val, ID_val = os_val.unsqueeze(1), ostime_val.unsqueeze(
        1), pfs_val.unsqueeze(1), pfstime_val.unsqueeze(1), ID_val.unsqueeze(1)
    os_test, ostime_test, pfs_test, pfstime_test, ID_test = os_test.unsqueeze(1), ostime_test.unsqueeze(
        1), pfs_test.unsqueeze(1), pfstime_test.unsqueeze(1), ID_test.unsqueeze(1)

    pred_train_save = torch.cat(
        (os_risk_train, os_train, ostime_train, pfs_risk_train, pfs_train, pfstime_train, ID_train, age_train), 1)
    pred_val_save = torch.cat(
        (os_risk_val, os_val, ostime_val, pfs_risk_val, pfs_val, pfstime_val, ID_val, age_val), 1)
    pred_test_save = torch.cat(
        (os_risk_test, os_test, ostime_test, pfs_risk_test, pfs_test, pfstime_test, ID_test, age_test), 1)

    pred_train_save = pred_train_save.cpu().numpy()
    pred_val_save = pred_val_save.cpu().numpy()
    pred_test_save = pred_test_save.cpu().numpy()

    pred_train_save = pd.DataFrame(pred_train_save)
    pred_val_save = pd.DataFrame(pred_val_save)
    pred_test_save = pd.DataFrame(pred_test_save)

    pred_train_save.to_csv(cfg['name'] + "_training.csv")
    pred_val_save.to_csv(cfg['name'] + "_validation.csv")
    pred_test_save.to_csv(cfg['name'] + "_testing.csv")

    # If only inference is required on test set:
    model.load_state_dict(torch.load(f"{cfg['name']}_PFS.pth"))
    _, pfs_risk = model_run_gpu(model, test_files, val_tf)
    model.load_state_dict(torch.load(f"{cfg['name']}_OS.pth"))
    os_risk, _ = model_run_gpu(model, test_files, val_tf)

    pred_save = torch.cat((os_risk, pfs_risk), 1)
    pred_save = pred_save.cpu().numpy()
    pred_save = pd.DataFrame(pred_save)
    pred_save.to_csv({cfg['name']} + "_TCIA.csv")


if __name__ == "__main__":
    main()
