import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from functions.loss import loss
from param_config.param_config import *
from functions.save_model import save_model
from functions.train_one_epoch import train_one_epoch
from functions.evaluate import evaluate
from functions.create_optimizer import create_optimizer
from functions.load_data import load_data
from functions.create_model import create_model
from functions.load_checkpoint import load_checkpoint



# This function creates a new directory for storing the output of the experiments.
def prepare_directory():
    # Get current date and time
    out_dir = f'/data1/temirlan/experiments/{input("Enter Direction To Save ->>> ")}'  # Format as 'mm_dd_yyyy_HH_MM_SS'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# This function saves the model's configuration parameters to a text file.
def save_config(out_dir):
    with open(os.path.join(out_dir, "config_model.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(config_dic) + "\n")

# This function trains and evaluates the model.
#{1
def train_and_evaluate(out_dir, net, optimizer, device, data_loader_train, data_loader_val, EPOCH_NUM, COMP_TOP, COMP_CNCT, COMP_TAE, TAU, checkpoint_number):
    criterion = loss(comp_top=COMP_TOP, comp_cnct=COMP_CNCT,comp_tae=COMP_TAE, tau=TAU, lmda=lmda1, lmda2=lmda2, lmda3=lmda3)    # Defining the loss function
#1}
    loss_scaler = NativeScaler()  # Defining the loss scaler
    unique_subdir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if USE_CHECKPOINT: 
        log_writer = SummaryWriter(os.path.join(out_dir, unique_subdir), purge_step=checkpoint_number-1)  # Defining the log writer for tensorboard
    else:
        log_writer = SummaryWriter(os.path.join(out_dir, unique_subdir))  

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)  # Total number of trainable parameters

    # Start of the training loop
    for epoch in range(0, EPOCH_NUM):      
        # Train for one epoch and get the training statistics
        train_stats = train_one_epoch(net, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, None,
                                      None, log_writer=log_writer)
        # Save the model after each epoch
        epoch = int(save_model(output_dirnm=out_dir, model=net, model_without_ddp=net, optimizer=optimizer, loss_scaler=loss_scaler,
                   epoch=epoch))
        print(epoch)

        # Evaluate the model after each epoch and get the test statistics
        test_stats = evaluate(data_loader_val, criterion, net, device)

        # Logging the statistics
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        # Write the stats to the tensorboard
        #{
        # log_writer.add_scalars("Losses", {
        #     'val/val_loss': test_stats['loss'],
        #     'val/mse_loss': test_stats['mse'],
        #     'val/mmt_loss': test_stats['mmt'],
        #     'train/train_loss': train_stats['loss'],
        #     'train/mse_loss': train_stats['mse'],
        #     'train/mmt_loss': train_stats['mmt']
        # }, epoch)
        log_writer.add_scalars("Losses", {
            'val/val_loss': test_stats['loss'],
            'val/reconst_loss': test_stats['reconst'],
            'val/topo_loss': test_stats['topo'],
            'val/tae_loss': test_stats['tae'],
            'val/cnct_loss': test_stats['cnct'],
            'train/train_loss': train_stats['loss'],
            'train/reconst_loss': train_stats['reconst'],
            'train/topo_loss': train_stats['topo'],
            'train/tae_loss': train_stats['tae'],
            'train/cnct_loss': train_stats['cnct'],
        }, epoch)
        #}
        log_writer.flush()
        # Write the stats to a log file
        with open(os.path.join(out_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

# This function combines all steps of loading data, creating model, preparing directory, saving config, training and evaluation
def main():
    data_loader_train, data_loader_val = load_data()  # Loading the training and validation data
    net, device = create_model()  # Creating the model

    if USE_CHECKPOINT:
        net = load_checkpoint(net, CHECKPOINT_PATH)
        out_dir = '/'.join(CHECKPOINT_PATH.split('/')[:-1])
        checkpoint_string = CHECKPOINT_PATH.split('/')[-1]
        checkpoint_number = int(checkpoint_string.split('-')[-1].split('.')[0])
        print(checkpoint_number)
    else:
        out_dir = prepare_directory() # Preparing the output directory
        checkpoint_number = 0

    optimizer = create_optimizer(net)  # Creating the optimizer 
    save_config(out_dir)  # Saving the model's config
    train_and_evaluate(out_dir, net, optimizer, device, data_loader_train, data_loader_val, EPOCH_NUM, COMP_TOP, COMP_CNCT, COMP_TAE, TAU, checkpoint_number)

# If the script is run directly, call the main function
if __name__ == "__main__":
    main()
