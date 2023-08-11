# -*- coding: utf-8 -*-

# Import necessary packages
from deepcell_toolbox.metrics import Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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


def save_plot_to_file(nuc_pred, y_true, directory):
    """
    Save plot to specified directory.
    """
    filename = os.path.join(directory, 'sample_plot.png')

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

    plt.savefig(filename)
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

    # Get the unique results directory only once
    save_results_directory = get_unique_foldername("/data1/temirlan/results")
    os.makedirs(save_results_directory)  # Create the directory here

    save_object_based_statistics(df_nuc, save_results_directory)
    save_pixel_based_statistics(stats_df, save_results_directory)
    save_f1_scores_to_file(mean_f1, manual_f1, save_results_directory)
    
    # Saving plot to a file
    save_plot_to_file(nuc_pred, y_true, save_results_directory)


if __name__ == "__main__":
    main()
