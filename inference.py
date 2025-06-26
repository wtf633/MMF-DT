import torch
from monai.data import Dataset, DataLoader, list_data_collate
from typing import Tuple

from metrics import cox_log_rank, CIndex_lifeline


def _build_loader(files, transforms, batch_size):
    return DataLoader(
        Dataset(files, transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )


def model_eval_gpu(model,
                   data_files,
                   transforms,
                   batch_size: int = 16):
    loader = _build_loader(data_files, transforms, batch_size)

    with torch.no_grad():
        model.eval()
        all_os_evt = all_os_time = all_pfs_evt = all_pfs_time = None
        all_os_out = all_pfs_out = all_id = all_age_out = None

        for step, batch in enumerate(loader):
            inputs = batch["input"].cuda()
            os_evt = batch["OS_status"].cuda()
            os_time = batch["OS_time"].cuda()
            pfs_evt = batch["PFS_status"].cuda()
            pfs_time = batch["PFS_time"].cuda()
            id_tensor = torch.tensor(
                [int(i) for i in batch["ID"]]
            ).cuda()

            os_out, pfs_out, age_out, _ = model(inputs)

            if step == 0:
                all_os_evt, all_os_time = os_evt, os_time
                all_pfs_evt, all_pfs_time = pfs_evt, pfs_time
                all_os_out, all_pfs_out = os_out, pfs_out
                all_id, all_age_out = id_tensor, age_out
            else:
                all_os_evt = torch.cat([all_os_evt, os_evt])
                all_os_time = torch.cat([all_os_time, os_time])
                all_pfs_evt = torch.cat([all_pfs_evt, pfs_evt])
                all_pfs_time = torch.cat([all_pfs_time, pfs_time])
                all_os_out = torch.cat([all_os_out, os_out])
                all_pfs_out = torch.cat([all_pfs_out, pfs_out])
                all_id = torch.cat([all_id, id_tensor])
                all_age_out = torch.cat([all_age_out, age_out])

    os_p = cox_log_rank(all_os_out, all_os_evt, all_os_time)
    os_c = CIndex_lifeline(all_os_out, all_os_evt, all_os_time)
    pfs_p = cox_log_rank(all_pfs_out, all_pfs_evt, all_pfs_time)
    pfs_c = CIndex_lifeline(all_pfs_out, all_pfs_evt, all_pfs_time)

    print(
        f"\nModel evaluation"
        f"\nOS c-index: {os_c:.4f}   log-rank p: {os_p:.4e}"
        f"\nPFS c-index: {pfs_c:.4f}  log-rank p: {pfs_p:.4e}"
    )

    return (all_os_out, all_pfs_out,
            all_os_evt, all_os_time,
            all_pfs_evt, all_pfs_time,
            all_id, all_age_out)


def model_run_gpu(model,
                  data_files,
                  transforms,
                  batch_size: int = 100):
    loader = _build_loader(data_files, transforms, batch_size)

    with torch.no_grad():
        model.eval()
        all_os_out = all_pfs_out = None
        for step, batch in enumerate(loader):
            inputs = batch["input"].cuda()
            os_out, pfs_out, _, _ = model(inputs)
            if step == 0:
                all_os_out, all_pfs_out = os_out, pfs_out
            else:
                all_os_out = torch.cat([all_os_out, os_out])
                all_pfs_out = torch.cat([all_pfs_out, pfs_out])
    return all_os_out, all_pfs_out
