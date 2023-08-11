from topo_cell_seg.swinunet_transform.swinunet import SwinUnet
import torch
from torchinfo import summary

def create_model():
    net = SwinUnet(num_classes=1)
    out = net(torch.randn(10, 2, 256, 256))
    device = torch.device('cuda')
    net.to(device)

    print("to {}".format(device))

    summary(net, (10, 2, 256, 256))

    return net, device