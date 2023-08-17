# -*- coding: utf-8 -*-

# Import necessary packages
from deepcell_toolbox.metrics import Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gudhi as gd
from ripser import ripser
from scipy import sparse


def load_predictions_and_labels():
    """
    Loads prediction and label numpy arrays from the file system.
    """
    nuc_pred = np.load('./tmp_pred/nucpred.npy')
    y_true = np.load('./tmp_pred/ytrue.npy')

    return nuc_pred, y_true


def calculate_metrics(pred, true):
    """
    Calculates the metrics for the given predictions and labels.
    """
    metrics = Metrics('Nuc stats 2', seg=False)
    df_nuc = metrics.calc_object_stats(pred.astype(int), true.astype(int))
    stats_list = metrics.calc_pixel_stats(pred.astype(int), true.astype(int))

    return df_nuc, pd.DataFrame(stats_list)


def save_object_based_statistics(df_nuc, directory):
    filename = os.path.join(directory, 'object_based_statistics.txt')
    with open(filename, 'w') as f:
        f.write("____________Object-based statistics____________\n\n")

        f.write(f"Number of true cells: {df_nuc['n_true'].sum()}\n")
        f.write(f"Number of predicted cells: {df_nuc['n_pred'].sum()}\n\n")

        print(df_nuc)

        correct = df_nuc['correct_detections'].sum()
        incorrect = df_nuc['n_pred'].sum() - correct
        recall = correct / df_nuc['n_true'].sum()
        precision = correct / df_nuc['n_pred'].sum()
        jaccard = df_nuc['jaccard'].mean()


        f.write(f"Correct detections:  {correct}     Recall: {recall:.4%}\n")
        f.write(f"Incorrect detections: {incorrect}     Precision: {precision:.4%}\n\n")
        f.write(f"Average Pixel IOU (Jaccard Index): {jaccard}")

        # Add other statistics as required...


def plot_random_sample(nuc_pred, y_true):
    """
    Plots a random sample from the predicted and true labels.
    """
    idx = np.random.randint(1324)

    plt.figure(figsize=(14, 14))

    plt.subplot(121)
    plt.axis('off')
    plt.title('Predicted Nuc')
    plt.imshow(nuc_pred[idx, ...])

    plt.subplot(122)
    plt.axis('off')
    plt.title('True Nuc')
    plt.imshow(y_true[idx, ...])

    plt.show()

def getf1(pc, rc):
    return 2/(1/pc + 1/rc)

def get_unique_foldername(directory):
    """
    Get a unique folder name in the given directory.
    """
    while True:
        foldername = os.path.join(directory, input('Type Name for the folder ->>> '))
        if not os.path.exists(foldername):
            return foldername
        else:
            raise TypeError

def save_pixel_based_statistics(stats_df, directory):
    filename = os.path.join(directory, 'pixel_based_statistics.txt')
    with open(filename, 'w') as f:
        f.write("____________Pixel-based statistics____________\n\n")
        f.write(stats_df.to_string())

def save_results_to_file(df_nuc, stats_df, directory):
    """
    Save results to specified directory (no longer unique).
    """
    nuc_file = os.path.join(directory, 'nuc_stats.csv')
    stats_file = os.path.join(directory, 'stats.csv')

    df_nuc.to_csv(nuc_file)
    stats_df.to_csv(stats_file)

    print(f'Results saved to {nuc_file} and {stats_file}')


def save_plot_to_file(nuc_pred, y_true, directory, idx):
    """
    Save plot to specified directory.
    """
    filename = os.path.join(directory, 'sample_plot.png')

  
    plt.figure(figsize=(14, 14))

    plt.subplot(121)
    plt.axis('off')
    plt.title('Predicted Nuc')
    plt.imshow(nuc_pred[idx, ...])

    plt.subplot(122)
    plt.axis('off')
    plt.title('True Nuc')
    plt.imshow(y_true[idx, ...])

    plt.savefig(filename, bbox_inches='tight', dpi=1000)
    plt.close()

    print(f'Plot saved to {filename}')


def save_f1_scores_to_file(auto_f1, manual_f1, directory):
    """
    Save F1 scores to a unique filename.
    """
    filename = os.path.join(directory, 'f1_scores.txt')

    with open(filename, 'w') as f:
        f.write(f"Mean F1 Score (Auto): {auto_f1}\n")
        f.write(f"Mean F1 Score (Manual): {manual_f1}\n")

    print(f'F1 scores saved to {filename}')

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
    print(sparseDM.shape)

    return ripser(sparseDM, distance_matrix=True, maxdim=1)["dgms"]

def plot_persistent_diagram(dgm, filename=None):
    """
    Plot the persistent diagram.
    
    If filename is provided, the plot will be saved to the filename.
    """


    print(dgm[0].shape)
    plt.figure(figsize=(7,7))
    dgm_0 = dgm[0]
    dgm_0[np.logical_not(np.isfinite(dgm_0))] = 1
    plt.scatter(dgm_0[:,0], dgm_0[:,1], marker= "+", label='0-dim features')

    dgm_1 = dgm[1]
    plt.scatter(dgm_1[:,0], dgm_1[:,1], marker= "x", label='1-dim features')

    plt.plot(np.linspace(0,1), np.linspace(0,1))
    plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=1000)
        plt.close()
    else:
        plt.show()
    
    return dgm_1

def plot_die_places(nuc_pred, dgm_1, filename=None):
    plt.figure()
    cp = nuc_pred.copy()
    for bth_v in dgm_1[:,0]:
        if bth_v == 0:
            continue
        cp[cp == bth_v] = 1
    plt.imshow(cp)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=1000)
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to load data, calculate metrics and plot a random sample.
    """
    nuc_pred, y_true = load_predictions_and_labels()
    df_nuc, stats_df = calculate_metrics(nuc_pred, y_true)
    print('\n')

    perc, rec = df_nuc['correct_detections'].sum()/df_nuc['n_pred'].sum(), df_nuc['correct_detections'].sum()/df_nuc['n_true'].sum()

    mean_f1 = stats_df[stats_df["name"] == "f1"]["value"].mean()

    print(f'Mean F1 Score (Auto): {mean_f1}')
    manual_f1 = getf1(perc, rec)
    print(f'Mean F1 Score (Manual): {manual_f1}')

    idx = np.random.randint(1324)
    dgm = lower_star_img(nuc_pred[idx])
    print(nuc_pred[idx])


    # Get the unique results directory only once
    save_results_directory = get_unique_foldername("/data1/temirlan/results")
    os.makedirs(save_results_directory)  # Create the directory here

    save_object_based_statistics(df_nuc, save_results_directory)
    save_pixel_based_statistics(stats_df, save_results_directory)
    save_f1_scores_to_file(mean_f1, manual_f1, save_results_directory)
    
    # Saving plot to a file
    save_plot_to_file(nuc_pred, y_true, save_results_directory, idx)
    
    dgm_1 = plot_persistent_diagram(dgm, filename=os.path.join(save_results_directory, 'persistent_diagram.png'))
    plot_die_places(nuc_pred[idx], dgm_1, filename=os.path.join(save_results_directory, 'die_fig.png'))


if __name__ == "__main__":
    main()

