import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import ot
from ot.lp import wasserstein_1d
import gudhi as gd
import numpy as np

from ripser import ripser
from scipy import sparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def lower_star_img(img):
    """
    Construct a lower star filtration on an image

    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data

    Returns
    -------
    I: ndarray (K, 2)
        A 0-dimensional persistence diagram corresponding to the sublevelset filtration
    """
    img = img.cpu().detach().numpy()
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))
    sparseDM = sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))

    return ripser(sparseDM, distance_matrix=True, maxdim=1)["dgms"]


def get_persistence_dim1(img):
    '''
    return 1d array
    '''
    # print(" sum img: ",torch.sum(img) )
    if torch.sum(img) == 0:
        return np.array([1.])
    else:
        dgm_1 = lower_star_img(img)[1]
        bn_ar = dgm_1[dgm_1[:, 1] > 0.9][:, 0]
        bn_ar = bn_ar[bn_ar > 0.0]
        # else:
        # print("bn_ar: ", bn_ar)
        return bn_ar


def loss_2_img(img_i, img_j):
    '''
    img_i: pred
    img_j: gt
    '''
    # print("img_i.shape  ",img_i.shape)
    # print(img_i.shape)
    pd_i_np, pd_j_np = get_persistence_dim1(img_i), get_persistence_dim1(img_j)
    # print(torch.tensor(pd_i_np))
    # print("Pd_i_np_1d:", pd_i_np_1d)
    if len(pd_i_np) == 0:
        pd_i_1d = torch.tensor([1.0]).to(device=device)
    else:
        pd_i_1d = torch.cat([img_i[(img_i == i)] for i in pd_i_np]).to(
            device=device)  ## pred has back_prop, retrieve all backward function from img_i

    if len(pd_j_np) == 0:
        pd_j_1d = torch.tensor([1.0]).to(device=device)
    else:
        pd_j_1d = torch.tensor(pd_j_np).to(device=device)  ## gt has no back_prop, use numpy

    # print("pd_i_1d", pd_i_1d)
    # print("Pd_i_np.shape : {}".format(pd_i_1d.shape[0]))
    a, b = (torch.ones(pd_i_1d.shape[0]) / pd_i_1d.shape[0]).to(device=device), (
                torch.ones(pd_j_1d.shape[0]) / pd_j_1d.shape[0]).to(device=device)
    # print("a" , a)
    try:
        loss = wasserstein_1d(
            pd_i_1d,
            pd_j_1d,
            a,
            b
        )
        return loss
    except Exception as e:
        print("pd_i_1d", pd_i_1d)
        print("pd_j_1d", pd_j_1d)


class topo_loss(nn.Module):
    
    def __init__(self, comp_top, lmda=5e-2):
        super(topo_loss, self).__init__()
        self.comp_mse = MSELoss()
        self.lmda = lmda
        self.comp_top = comp_top

    def comp_topoloss(self, inp, targ):
        # print("Input Shape: ",inp.shape)
        losses = []
        for image_pair in zip(inp, targ):
            image_1, image_2 = image_pair
            # Compute loss for this image pair
            image_1, image_2 = image_1.squeeze(), image_2.squeeze()
            loss = loss_2_img(image_1, image_2)
            # print("2_img_loss: ",loss)
            # Append loss to list of losses
            losses.append(loss)
        # Compute mean of losses for the batch
        batch_loss = torch.mean(torch.stack(losses))
        return batch_loss

    def forward(self, outs, targets):
        mse_loss = self.comp_mse(outs, targets)
        if self.comp_top:
            topo_loss = self.comp_topoloss(outs, targets)
        else:
            topo_loss = 0
        # topo_loss = torch.tensor([0]).to(device=device)

        return (mse_loss + self.lmda * topo_loss,
                mse_loss.item(),
                topo_loss
                )


if __name__ == "__main__":
    # test_metric = topo_loss()

    nucinnd = np.load('../imgs/train/2452_3/nucinnd.npy')
    nucinnd_1 = np.load('../imgs/train/2452_3/nucinnd.npy')
    nucinnd.shape
    pred = torch.Tensor(nucinnd).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
    pred.requires_grad_()
    # pred[:,:,50,78] = 0.8
    # pred[:,:,50,77] = 0.7
    # pred[:,:,50,76] = 0.6
    gtrth = torch.Tensor(nucinnd_1).unsqueeze(dim=0).unsqueeze(dim=0).cuda()

    Tlss = topo_loss()
    los_val = Tlss.forward(pred, gtrth)

    print(
        "The loss between the {} and {} is {}".format("Pred", "Gtruch", los_val)
    )