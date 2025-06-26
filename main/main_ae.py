import os
import torch
import pandas as pd

from data_utils import build_file_dicts, get_transforms
import models
from ae_engine import train_validate_test_ae, eval_ae
import argparse

pd.options.mode.chained_assignment = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    def parse_args():
        p = argparse.ArgumentParser(description="AutoEncoder survival model")
        # experiment identity
        p.add_argument("--name", default="multitask_subnet3", help="run tag / model prefix")
        p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
        # optimisation
        p.add_argument("--lr", type=float, default=1e-5)
        p.add_argument("--lr_decay", type=float, default=0.5)
        p.add_argument("--epochs", type=int, default=250)
        p.add_argument("--drop_rate", type=float, default=0.25)
        p.add_argument("--rand_p", type=float, default=0.55)
        p.add_argument("--skip_epoch_model", type=int, default=50)
        # batch sizes
        p.add_argument("--train_bs", type=int, default=16)
        p.add_argument("--val_bs", type=int, default=8)
        p.add_argument("--test_bs", type=int, default=8)
        # paths
        p.add_argument("--train_img_dir", required=True)
        p.add_argument("--val_img_dir", required=True)
        p.add_argument("--test_img_dir", required=True)
        p.add_argument("--csv_root", required=True,
                       help="folder containing train_events.csv, valid_events.csv, test_events.csv")
        return p.parse_args()

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = vars(args)

    train_files = build_file_dicts(
        cfg['train_img_dir'], f"{cfg['csv_root']}/train_events.csv"
    )
    val_files = build_file_dicts(
        cfg['val_img_dir'], f"{cfg['csv_root']}/valid_events.csv"
    )
    test_files = build_file_dicts(
        cfg['test_img_dir'], f"{cfg['csv_root']}/test_events.csv"
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    model = train_validate_test_ae(
        train_files,
        val_files,
        test_files,
        cfg,
        device,
    )

    # ------------------------------------------------------------
    # Evaluation with best checkpoints
    # ------------------------------------------------------------

    model.load_state_dict(torch.load(f"{cfg['name']}_PFS.pth"))
    _, pfs_risk_train, os_train, ostime_train, pfs_train, pfstime_train, ID_train, age_train, bn_feature_train = eval_ae(
        model.cuda(),
        train_files,
        get_transforms(cfg["rand_p"])[1])
    _, pfs_risk_val, os_val, ostime_val, pfs_val, pfstime_val, ID_val, age_val, bn_feature_val = eval_ae(
        model.cuda(), val_files,
        get_transforms(cfg["rand_p"])[1])
    _, pfs_risk_test, os_test, ostime_test, pfs_test, pfstime_test, ID_test, age_test, bn_feature_test = eval_ae(
        model.cuda(),
        test_files,
        get_transforms(cfg["rand_p"])[1])


    model.load_state_dict(torch.load(f"{cfg['name']}_OS.pth"))
    os_risk_train, _, _, _, _, _, _, _, _ = eval_ae(model, train_files, get_transforms(cfg["rand_p"])[1])
    os_risk_val, _, _, _, _, _, _, _, _ = eval_ae(model, val_files, get_transforms(cfg["rand_p"])[1])
    os_risk_test, _, _, _, _, _, _, _, _ = eval_ae(model, test_files, get_transforms(cfg["rand_p"])[1])

    os_train, ostime_train, pfs_train, pfstime_train, ID_train = os_train.unsqueeze(1), ostime_train.unsqueeze(
        1), pfs_train.unsqueeze(1), pfstime_train.unsqueeze(1), ID_train.unsqueeze(1)
    os_val, ostime_val, pfs_val, pfstime_val, ID_val = os_val.unsqueeze(1), ostime_val.unsqueeze(
        1), pfs_val.unsqueeze(1), pfstime_val.unsqueeze(1), ID_val.unsqueeze(1)
    os_test, ostime_test, pfs_test, pfstime_test, ID_test = os_test.unsqueeze(1), ostime_test.unsqueeze(
        1), pfs_test.unsqueeze(1), pfstime_test.unsqueeze(1), ID_test.unsqueeze(1)

    # save the model prediction risk
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

    # save the bottleneck features
    pred_f_train = torch.cat((ID_train, bn_feature_train), 1)
    pred_f_val = torch.cat((ID_val, bn_feature_val), 1)
    pred_f_test = torch.cat((ID_test, bn_feature_test), 1)

    pred_f_train = pred_f_train.cpu().numpy()
    pred_f_val = pred_f_val.cpu().numpy()
    pred_f_test = pred_f_test.cpu().numpy()

    pred_f_train = pd.DataFrame(pred_f_train)
    pred_f_val = pd.DataFrame(pred_f_val)
    pred_f_test = pd.DataFrame(pred_f_test)

    pred_f_train.to_csv(cfg['name'] + "_feature_train.csv")
    pred_f_val.to_csv(cfg['name'] + "_feature_val.csv")
    pred_f_test.to_csv(cfg['name'] + "_feature_test.csv")

    model.load_state_dict(torch.load(cfg['name'] + "_PFS.pth"))
    _, pfs_risk, bn_feature = eval_ae(model, test_files, get_transforms(cfg["rand_p"])[1])

    model.load_state_dict(torch.load(cfg['name'] + "_OS.pth"))
    os_risk, _, _ = eval_ae(model, test_files, get_transforms(cfg["rand_p"])[1])
    pred_save = torch.cat((os_risk, pfs_risk), 1)
    pred_save = pred_save.cpu().numpy()
    pred_save = pd.DataFrame(pred_save)
    pred_save.to_csv(cfg['name'] + "_TCIA.csv")

    bn_feature = bn_feature.cpu().numpy()
    bn_feature = pd.DataFrame(bn_feature)
    bn_feature.to_csv(cfg['name'] + "_feature_TCIA.csv")


if __name__ == "__main__":
    main()
