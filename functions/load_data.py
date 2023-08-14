from param_config.param_config import *
from swinunet_transform.tissuenetdata import SeprtSeg
import torch

def load_data():
    dataset_train = SeprtSeg(root_dir = '../imgs/train')
    dataset_val = SeprtSeg(root_dir = '../imgs/val')
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )
    return data_loader_train, data_loader_val