
from swinunet_transform.swinunet import SwinUnet
import torch
from torchinfo import summary
import argparse
from swinunet_transform.config import *

# Initialize argparse object
args = argparse.Namespace(
    cfg='../topo_cell_seg/swinunet_transform/swin_tiny_patch4_window7_224.yaml',
    batch_size=None,
    zip=False,
    cache_mode=None,
    resume=None,
    accumulation_steps=None,
    use_checkpoint=False,
    amp_opt_level=None,
    tag=None,
    eval=False,
    throughput=False,
    opts=[]  
)

# Get the config object
cfg = get_config(args)

def create_model():
    net = SwinUnet(num_classes=1)
    net.load_from(cfg)
    out = net(torch.randn(10, 2, 256, 256))
    device = torch.device('cuda')
    net.to(device)

    print("to {}".format(device))

    summary(net, (10, 2, 256, 256))

    return net, device