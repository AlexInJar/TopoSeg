import os
import ast

USE_CHECKPOINT = ast.literal_eval(os.getenv('USE_CHECKPOINT'))  # Set this variable to True to use a checkpoint
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH')
EPOCH_NUM = int(os.getenv('EPOCH_NUM'))
COMP_TOP = ast.literal_eval(os.getenv('COMP_TOP'))
COMP_CNCT = ast.literal_eval(os.getenv('COMP_CNCT'))
COMP_TAE = ast.literal_eval(os.getenv('COMP_TAE'))
TAU = float(os.getenv('TAU'))

lr = 1e-3* 32/256
min_lr = 1e-6
weight_decay = 0.05
layer_decay = 0.75
lmda1 = 5e-1 #topo_loss
lmda2 = 2e-5 #tae_loss
lmda3 = 8.77e-7 #conn_loss


batch_size = 8
num_workers = 10

pin_mem = True

config_dic = {
    'lr' : 1e-3* 32/256,
    'min_lr' : 1e-6,
    'weight_decay' : 0.05,
    'layer_decay' : 0.75,
    'topo_loss' : lmda1,
    'tae_loss' : lmda2,
    'conn_loss' : lmda3
}