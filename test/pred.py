# -*- coding: utf-8 -*-

# Import necessary libraries
import os
import numpy as np
import torch
from topo_cell_seg.swinunet_transform.swinunet import SwinUnet
from topo_cell_seg.swinunet_transform.tissuenetdata import SeprtSeg
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def setup_environment():
    """
    Set up the CUDA environment.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def load_model():
    """
    Load the model.
    """
    model = SwinUnet(num_classes=1)
    chkpt_dir = '/data1/temirlan/experiments/nuc_test_07_26_2023_21_48_33/checkpoint-249.pth'
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    return model


def create_dataset():
    """
    Create and return the test dataset.
    """
    non_transf = lambda x: x

    dataset_test = SeprtSeg(
        root_dir = '../imgs/test',
        transform = non_transf,
        load_inseg = True
    )

    dataset_test.paths = ["{}".format(pthi) for pthi in dataset_test.paths]
    dataset_test.tis_dic = {"{}".format(ki): v_ for ki, v_ in dataset_test.tis_dic.items()}

    return dataset_test


def load_data(dataset_test):
    """
    Load the test data.
    """
    Xtest = np.zeros((len(dataset_test), 256, 256, 2))
    ytest = np.zeros((len(dataset_test), 256, 256, 2))

    for idx in tqdm(range(len(dataset_test))):
        Xtest[idx,...] = dataset_test.__getitem__(idx)['image']
        ytest[idx,...] = dataset_test.__getitem__(idx)['insnm']

    return Xtest, ytest


def run_model(model, Xtest, dataset_test):
    """
    Run the model to get predictions.
    """
    X = torch.Tensor(Xtest)
    X = torch.einsum('nhwc->nchw', X).cuda()

    X = X.detach().cpu()

    celres, nucres = np.zeros((len(dataset_test), 256, 256)), np.zeros((len(dataset_test), 256, 256))

    for i in tqdm(range(1324)):
        nuc_ind = model(X[i:(i+1),...])
        nucres[i,...] = nuc_ind.detach()

    return celres, nucres, X


def watershed_postprocessing(nucres):
    """
    Apply watershed segmentation on the predictions.
    """
    def wtrshed(gryimg, maskthres=0.08):
        coords = peak_local_max(gryimg, footprint=np.ones((9, 9)), labels = ( gryimg > maskthres))
        mask = np.zeros(gryimg.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        label_img = watershed(-1 * gryimg, markers,
                                    mask=gryimg > maskthres,
                                    watershed_line=0)
        return label_img

    def wtrshed_all(stacked_gryimg, channel = 2):
        toret = np.zeros(stacked_gryimg.shape)
        for i, img in tqdm(enumerate(stacked_gryimg)):
            toret[i,...] = wtrshed(img)
        return toret

    return wtrshed_all(nucres)


def save_results(nuc_pred, ytest, X):
    """
    Save the predictions and the original data.
    """

    # Create directory if it does not exist
    if not os.path.exists('./tmp_pred'):
        os.makedirs('./tmp_pred')

    np.save('./tmp_pred/nucpred.npy', nuc_pred)
    np.save('./tmp_pred/ytrue.npy', ytest[...,1])
    np.save('./X.npy', torch.einsum('nchw->nhwc', X).detach().numpy())


def main():
    """
    Main function to set up environment, load model, create dataset, load data,
    run model, apply post-processing, and save results.
    """
    setup_environment()
    model = load_model()
    dataset_test = create_dataset()
    Xtest, ytest = load_data(dataset_test)
    celres, nucres, X = run_model(model, Xtest, dataset_test)
    nuc_pred = watershed_postprocessing(nucres)
    save_results(nuc_pred, ytest, X)


if __name__ == "__main__":
    main()
