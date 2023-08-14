import torch
from util.misc import is_main_process

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)