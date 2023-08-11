import torch
from topo_cell_seg.util.misc import is_main_process

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)