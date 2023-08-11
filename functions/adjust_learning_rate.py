import math
from topo_cell_seg.param_config.param_config import *

def adjust_learning_rate(optimizer, epoch, num_epochs=250, warmup_epochs=10, lr=lr, min_lr=min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # print(num_epochs)
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr