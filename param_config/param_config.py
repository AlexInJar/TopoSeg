import os
import ast

USE_CHECKPOINT = ast.literal_eval(os.getenv('USE_CHECKPOINT'))  # Set this variable to True to use a checkpoint
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH')
EPOCH_NUM = int(os.getenv('EPOCH_NUM'))
COMP_TOP = ast.literal_eval(os.getenv('COMP_TOP'))

lr = 1e-3* 32/256
min_lr = 1e-6
weight_decay = 0.05
layer_decay = 0.75

batch_size = 8
num_workers = 10

pin_mem = True

config_dic = {
    'lr' : 1e-3* 32/256,
    'min_lr' : 1e-6,
    'weight_decay' : 0.05,
    'layer_decay' : 0.75
}