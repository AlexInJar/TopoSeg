# Project Description

This project uses a deep learning approach for cell segmentation, leveraging a number of functions and modules. The main script is responsible for orchestrating the various steps involved in training and evaluating the model.

## Main Script

The script main.py handles the entire pipeline of training and evaluation. Here's a step-by-step overview of the functionalities provided by the main script:

    Load Data: The load_data() function is responsible for loading the training and validation datasets. The function returns two data loaders, one for training data and one for validation data.
    
    Create Model: The create_model() function is used to create the neural network model that will be used for training. This function returns the model and the device (CPU or GPU) on which computations will be performed.
    
    Create Optimizer: The create_optimizer(net) function creates the optimizer that will be used to optimize the neural network. It takes the neural network as an input and returns the optimizer.
    
    Prepare Directory: The prepare_directory() function creates a directory where the outputs of the experiments will be stored.
    
    Save Configuration: The save_config(out_dir) function saves the configuration parameters of the model into a text file in the output directory.

    Train and Evaluate: The train_and_evaluate() function is the core of the script. It performs the following tasks:
            Initializes the loss function and loss scaler.
            Defines a log writer for TensorBoard.
            Initiates the training loop and trains the model for a specified number of epochs.
            After each epoch, it saves the model and evaluates it on the validation set.
            It logs the statistics of training and evaluation to both a text file and TensorBoard.

The main() function in the script orchestrates these steps. It first loads the data, creates the model and optimizer, prepares the output directory, saves the model configuration, and finally trains and evaluates the model.

If you run this script directly, it will initiate the above-mentioned process by calling the main() function.

Note: You may need to adjust the paths and other parameters in the script to suit your own data and system setup.

## Function Descriptions

Descriptions of functions located in the folder functions.

### adjust_learning_rate() 

File -> _adjust_learning_rate.py_

Located in the functions folder, the function adjust_learning_rate() is a crucial part of the training process, responsible for adjusting the learning rate of the optimizer as the training progresses.

The adjust_learning_rate() function follows a specific policy to adjust the learning rate of the optimizer during the training process. This is done to ensure that the model converges faster and avoids overshooting the minimum of the loss function. Here's a breakdown of the arguments:

    optimizer: The optimizer whose learning rate needs to be adjusted.
    epoch: The current epoch number.
    num_epochs=250: The total number of epochs the training process will run for. The default is 250.
    warmup_epochs=10: The number of epochs that will be used for warming up the training. During this period, the learning rate will linearly increase from 0 to its initial value. The default is 10.
    lr=lr: The initial learning rate before adjusting. This comes from the imported config file.
    min_lr=min_lr: The minimum learning rate after adjusting. This comes from the imported config file.

In the first warmup_epochs, the learning rate increases linearly from 0 to its initial value. After the warmup period, the learning rate is adjusted using a half-cycle cosine decay schedule. This means it smoothly decreases and increases between the min_lr and lr in a cosine pattern.

Finally, the function applies the adjusted learning rate to each parameter group in the optimizer. If a lr_scale is defined in the parameter group, it's multiplied with the learning rate.

This function returns the adjusted learning rate.

The adjustment of learning rate during training can help to enhance the performance of the model and reduce the training time.

To use this function, simply call it at the beginning of each epoch in the training loop. It will automatically adjust the learning rate of the optimizer based on the current epoch.

Example of usage:

lr = adjust_learning_rate(optimizer, epoch, num_epochs, warmup_epochs, lr, min_lr)

Please note that this function needs to be called at each epoch for it to work properly.`

### create_model()

File -> _create_model.py_

Located in the functions folder, the function create_model() is used to create and initialize the model that will be used for training.

The create_model() function initializes an instance of the SwinUnet model, which is a variation of the U-Net model with Swin Transformers as the backbone. This model has been specifically designed for image segmentation tasks.

Here's a breakdown of the steps in the function:

    net = SwinUnet(num_classes=1): The function first initializes the SwinUnet model with one output class. This implies that the model will be performing binary segmentation.

    out = net(torch.randn(10, 2, 256, 256)): Next, it applies the model to a random batch of images to ensure the model is working as expected. The images are of shape (10, 2, 256, 256), meaning that there are 10 images in the batch, each with 2 channels (e.g., grayscale and mask) and dimensions 256x256.

    device = torch.device('cuda'): The function then defines the computation device. Here, it's set to use a GPU ('cuda'). If you don't have a GPU, you may need to change this to 'cpu'.

    net.to(device): The model is then transferred to the specified device.

    print("to {}".format(device)): A print statement confirms that the model has been transferred to the correct device.

    summary(net, (10, 2, 256, 256)): The summary() function from the torchinfo library is used to print a summary of the model, including the number of parameters and the output shape of each layer when applied to an input of shape (10, 2, 256, 256).

    return net, device: Finally, the function returns the model and the device on which the model will be running.

This function is used to create the model that will be trained. The function should be called before starting the training process.

Example of usage:

net, device = create_model()

Please note that the device (CPU or GPU) on which you want to run the computations should be available and properly configured in your system.

### param_groups_lrd()

File -> _create_optimizer.py_

The param_groups_lrd() function generates parameter groups for layer-wise learning rate decay. This approach is inspired by the BEiT model, where learning rate decay is applied differently at each layer of the model. The function creates parameter groups, with different learning rate scales and weight decays, which can be directly used by PyTorch optimizers.

The arguments for this function are:

    model: The model for which the parameter groups are being created.
    weight_decay: The weight decay value for layers where weight decay is applied.
    no_weight_decay_list: A list of model parameters (by name) that should not have weight decay applied.
    layer_decay: The rate at which learning rate scales decrease across layers.

The function first calculates the learning rate scales for each layer. It then goes through each parameter in the model and assigns it to a group based on its layer and whether it requires weight decay. These groups are then returned as a list of dictionaries, each representing a parameter group with its learning rate scale, weight decay, and parameters.

### create_optimizer()

File -> _create_optimizer.py_

The create_optimizer() function creates the optimizer that will be used to optimize the neural network. It first calls the param_groups_lrd() function to get the parameter groups with layer-wise learning rate decay. These parameter groups are then provided to the AdamW optimizer from the PyTorch library, along with the initial learning rate. The optimizer is then returned.

The create_optimizer() function is used to create the optimizer that will be used in the training process. It should be called before starting the training process.

Example of usage:

optimizer = create_optimizer(net)

Please note that the learning rate, layer decay, and weight decay values come from an imported config file and may need to be adjusted based on your specific needs.

### evaluate()

File -> _evaluate.py_

The evaluate() function runs the model in evaluation mode on a provided dataset and calculates the loss and other metrics.

Here's a breakdown of the steps in the function:

    model.eval(): Sets the model in evaluation mode. In this mode, the network behaves differently from training mode in terms of dropout and batch normalization.

    for batch in metric_logger.log_every(data_loader, 10, header): Iterates over batches of images and corresponding target values in the provided data loader. The MetricLogger object logs the process every 10 batches.

    images = images.to(device, non_blocking=True): Transfers the images to the computation device (CPU or GPU). The non_blocking=True argument means that the data transfer will not block the whole process and allows for asynchronous GPU copies.

    with torch.cuda.amp.autocast(): This context manager enables the automatic mixed precision (AMP) feature in PyTorch. AMP speeds up computation and reduces the memory usage while maintaining the model's accuracy.

    output = model(images): Passes the batch of images through the model, obtaining the model's output (predictions).

    loss, mse_v, mmt_v = criterion(output, targets): Calculates the loss and additional metrics (in this case, mean squared error and an additional metric represented by mmt_v) using the model's output and the target values.

    metric_logger.update(...): Updates the MetricLogger with the newly calculated metrics.

    metric_logger.synchronize_between_processes(): Synchronizes the metrics between different processes. This step is useful when using multiple GPUs.

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}: Returns a dictionary with the global average of all logged metrics.

This function is used during the validation phase of the training loop and can be used to assess the model's performance on any dataset.

Example of usage:

test_stats = evaluate(data_loader_val, criterion, net, device)

Please note that the device (CPU or GPU) on which you want to run the computations should be available and properly configured in your system. Also, the data_loader_val should be an instance of a PyTorch DataLoader, providing batches of images and target values. The criterion should be a PyTorch loss function suitable for your specific task.

### get_layer_id_for_resnext()

File -> _get_layer_id_for_resnext.py_

The get_layer_id_for_resnext() function identifies the layer to which a given parameter belongs. This is necessary for applying different learning rates or weight decay to parameters from different layers of the model.

The function takes as input:

    name: The name of the parameter.
    num_layers: The total number of layers in the model.

The function returns the ID of the layer to which the parameter belongs.

Here's a breakdown of how the function determines the layer ID:

    If the parameter name is 'cls_token' or 'pos_embed', the function returns 0, indicating that these parameters belong to the input layer.
    If the parameter name contains 'patch_embed', the function also returns 0, as these parameters are also part of the input layer.
    If the parameter name contains 'layer', the function splits the name by periods ('.') and uses the numbers in the second and third positions to determine the layer ID. In this case, it's assumed that the parameter name follows a certain pattern like 'layer.x.y', where 'x' and 'y' can be used to determine the layer ID.
    If the parameter name doesn't fall into any of the above categories, the function returns num_layers, indicating that these parameters belong to the output layer.

This function can be used for organizing parameters when using optimizers that apply different settings to parameters from different layers, such as in layer-wise learning rate decay.

Example of usage:

layer_id = get_layer_id_for_resnext(param_name, num_layers)

Please note that the param_name should be the name of a parameter in your model, and num_layers should be the total number of layers in your model. The function assumes a specific naming convention for the parameters; if your model uses a different convention, you may need to modify the function accordingly.

### get_layer_id_for_vit()

File -> _get_layer_id_for_vit.py_

The get_layer_id_for_vit() function maps each parameter to its respective layer ID in the ViT model. It is especially useful when different treatments are needed for parameters of different layers.

This function takes two inputs:

    name: The name of a parameter.
    num_layers: The total number of layers in the model.

The function then returns the ID of the layer to which the parameter belongs.

Here's a summary of the layer ID determination:

    If the parameter name is either 'cls_token' or 'pos_embed', the function returns 0, indicating that these parameters are associated with the input layer.
    If 'patch_embed' is present in the parameter name, the function also returns 0, suggesting these parameters belong to the input layer.
    If 'blocks' is in the parameter name, the function splits the name by periods ('.') and uses the third position to identify the layer ID. This implies the parameter name follows a pattern like 'blocks.x.y', where 'x' can be used to ascertain the layer ID.
    For any other parameters not categorized above, the function returns num_layers, suggesting these parameters are part of the output layer.

This function is particularly useful when using optimizers that require different settings for parameters from various layers, such as in layer-wise learning rate decay.

Example usage:

layer_id = get_layer_id_for_vit(param_name, num_layers)

Please note, param_name should be the name of a parameter in your model, and num_layers should be the total number of layers in your model. The function assumes a specific naming pattern for the parameters; if your model uses a different convention, you might need to adjust the function accordingly.

### load_data()

File -> _load_data.py_

The load_data() function loads and returns data loaders for the training and validation sets. The function utilizes PyTorch's DataLoader utility to efficiently load and handle the datasets.

This function does not take any input parameters and returns two DataLoaders: one for the training set and one for the validation set.

In more detail, the function operates as follows:

    Creates instances of the SeprtSeg class for both the training and validation sets. SeprtSeg is presumably a custom class defined in the tissuenetdata module within the swinunet_transform package. The function expects the training and validation images to be located in the '../imgs/train' and '../imgs/val' directories respectively.

    Creates data loaders for the training and validation datasets using PyTorch's DataLoader utility. The batch size, the number of worker processes, the use of pinned memory, and the option to drop the last incomplete batch are set based on the parameters defined in the param_config module.

    Returns the data loaders for the training and validation sets.

Here's an example of how you might use this function:

data_loader_train, data_loader_val = load_data()

You will then be able to use data_loader_train and data_loader_val in your training and validation loops respectively.

Please ensure that the directory paths passed to the SeprtSeg instances point to your actual training and validation datasets. Also, make sure that the SeprtSeg class and the parameters in param_config are appropriately defined for your specific use case.

### save_model()

File -> _save_model.py_

The save_model() function is used to save the current state of the model, optimizer, and the learning rate scaler at the end of an epoch. This allows for checkpointing during the training process, which is essential for recovery in case of failure, or for continuing training later.

The function takes in the following parameters:

    output_dirnm : The output directory where the checkpoints will be saved.
    epoch : The current epoch number.
    model : The model which is being trained.
    model_without_ddp : The model without the DistributedDataParallel wrapper. This is saved to make it easier to use the model in a non-distributed setting later.
    optimizer : The optimizer being used for training.
    loss_scaler : The learning rate scaler. This parameter is optional.
    output_dinmr : An additional output directory, possibly used for saving other information related to the model.

The function works as follows:

    If a loss scaler is provided, it creates a checkpoint dictionary containing the state dictionaries of the model, optimizer, and scaler, as well as the current epoch number. This checkpoint dictionary is then saved to a file in the specified output directory. The filename of the checkpoint contains the epoch number.
    If a loss scaler is not provided, it creates a dictionary only containing the current epoch number and saves a checkpoint of the model to the directory specified by output_dinmr. The filename of the checkpoint contains the epoch number.

Here's an example of how you might use this function:

save_model(output_dirnm='my_dir', epoch=epoch, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, output_dinmr='my_dir')

Please ensure that the directory paths passed to the function actually exist to prevent any errors during runtime.

### save_on_master()

File -> _save_on_master.py_

The save_on_master() function saves the provided arguments using PyTorch's torch.save() function, but only if the current process is the main process.

The function takes arbitrary arguments and keyword arguments, which are directly passed to torch.save(). This can be any data that PyTorch's torch.save() function supports, such as model states, optimizer states, or general-purpose data.

The function operates by using the utility function is_main_process(), which checks whether the current process is the main process. If it is, torch.save() is used to save the provided data. If it's not the main process, the function does nothing.

Here's an example of how you might use this function:

model_state = model.state_dict()
save_on_master(model_state, 'model_state.pth')

In this example, the state of model will be saved to model_state.pth, but only if the current process is the main process.

Please note that the directory in which you wish to save the file must exist, otherwise torch.save() will raise an error.

### topo_loss()

File -> _tp_loss.py_

This class extends PyTorch's nn.Module and can be used as a loss function during training. The forward() method computes the MSE loss and topological loss between a pair of images, and returns a weighted sum of these losses. It also provides a separate method comp_topoloss() to compute the topological loss between a pair of images.

In the final part of the script, a topo_loss object is created and used to compute the loss between a pair of example images. This part of the script is only run when the script is executed as a standalone script, and not when it is imported as a module.

The script located in the functions directory provides several functions and a topo_loss class to compute topological loss based on persistence diagrams. This is a specialized form of loss used in scenarios like topology-aware segmentation where the aim is to make sure that the learned segmentation respects certain topological properties.

The functions in this script use several libraries, such as gudhi for topological data analysis, ot for optimal transport, and numpy and torch for general numerical computing.

The main class in the script is topo_loss, which extends PyTorch's nn.Module. This class can be used as a loss function during training of deep learning models. It computes both a mean squared error (MSE) loss and a topological loss, and returns a weighted sum of these losses.

Here is a summary of the functions and class:

### lwst_simplex(img)

File -> _tp_loss.py_

This function takes an image as input and computes the lower star simplex filtration. This filtration is a technique from topological data analysis that encodes information about the topology of the image. The output is a SimplexTree object from the gudhi library, which represents the filtration.

### comp_persis(img)

File -> _tp_loss.py_

This function takes an image as input, computes the lower star simplex filtration using the previous function, and extends this filtration to an extended persistence diagram. This diagram represents the topological features of the image in a form that is easy to compare between different images.

### extract_second_from_tuples(lst)

File -> _tp_loss.py_

This function takes a list of tuples as input, where each tuple represents a topological feature from a persistence diagram. The function extracts the second value from the tuples, which represents the death value of the feature. It returns an array of these death values for all features with death value greater than 0.9.

### get_persistence_diget_persistence_dim1m1(img)

File -> _tp_loss.py_

This function takes an image as input, computes its extended persistence diagram using the comp_persis() function, and extracts the death values of the topological features using the extract_second_from_tuples() function.

### loss_2_img(img_i, img_j)

File -> _tp_loss.py_

This function takes two images as input, computes their persistence diagrams, and then computes the Wasserstein distance between these diagrams. This distance is a measure of the difference between the topological features of the two images.

### train_one_epoch()

File -> _train_one_epoch.py_

This script located in the functions directory contains a function train_one_epoch for training a machine learning model for one epoch. The function is quite general and can be used with any PyTorch model and dataset. It also provides support for mixed precision training and gradient accumulation, as well as various forms of logging and metrics.

Here is a breakdown of the main components of the function:

This function accepts a PyTorch model, a loss function, a data loader, an optimizer, a device, the current epoch number, a loss scaler for mixed precision training, a max norm for gradient clipping, a Mixup function for data augmentation, a log writer for tensorboard, and any other additional arguments.

The function starts by setting the model to train mode and initializing a MetricLogger object from the misc module. This object is used to log various metrics such as the learning rate and the loss.

The function then enters a main loop where it iterates over the data loader. For each batch of data, it applies the Mixup function if provided, passes the data through the model, computes the loss, and performs backpropagation.

During backpropagation, the loss is scaled by the loss scaler, and the gradients are clipped to the provided max norm. If gradient accumulation is used, the gradients are only updated after a certain number of iterations, and the optimizer is only stepped and zeroed out after this number of iterations.

After the loss is backpropagated, the function synchronizes the CUDA operations and updates the metrics in the MetricLogger. It also updates the learning rate in the optimizer and logs the loss and learning rate to tensorboard.

At the end of the function, the metrics are synchronized across all processes, and the function prints out the averaged metrics and returns them in a dictionary.

The script also includes an initial loop that prints out the keys of the items in the data loader. This loop is intended to be used for debugging purposes and can be removed in a production environment.

In summary, this script contains a flexible and powerful training function that can be used for training a wide variety of PyTorch models.

## Configuration Parameters Descriptions

Descriptions of function located in the folder param_config.

### param_config()

File -> _param_config.py_

This script, located in the param_config directory, is responsible for setting hyperparameters and configuration values for a machine learning model. The variables set here often control various aspects of the training process, including learning rate, weight decay, and batch size.

Here is a brief explanation of what each of these parameters signifies:

    Learning rate (lr): This value represents the step size at which the model learns. A smaller learning rate means that the model will learn slowly, while a larger learning rate allows the model to learn quickly. Here, it is computed as 1e-3 * 32 / 256.

    Minimum learning rate (min_lr): This is the smallest value that the learning rate can reach during training. It is often used in conjunction with learning rate scheduling strategies that reduce the learning rate over time.

    Weight decay (weight_decay): This is a regularization technique that discourages large weights in the model by adding a penalty (proportional to the weight size) to the loss function. Weight decay can help prevent overfitting.

    Layer decay (layer_decay): This value is used to reduce the learning rate for deeper layers in the network. It is typically used when the network is very deep and the features learned by the deeper layers are more abstract and less sensitive to changes.

    Batch size (batch_size): This is the number of samples processed before the model's internal parameters are updated.

    Number of workers (num_workers): This parameter is used when loading data in parallel using PyTorch's DataLoader. It specifies how many subprocesses to use for data loading. The more workers, the smoother the data loading process will be.

    Pin Memory (pin_mem): If pin_mem=True, it will be copied into the CUDA pinned memory, which enables faster data transfer to the GPU.

Additionally, this script creates a dictionary named config_dic that contains the same information in a format that's easier to pass around between different functions or modules. Storing hyperparameters in a dictionary is a common practice as it allows easy access to all the different parameters throughout the codebase.

The values set here are specific to the task at hand and may require adjustment based on the nature of the data, the complexity of the model, or the specific goals of the training process.

## Swinunet Transform and Util Descriptions 

### nptransform.py
This module contains functions related to transformations on NumPy arrays. Transformations might include rotations, scaling, cropping, normalizations, and more. These transformations are designed to augment the data, improving the model's ability to generalize from the training data to unseen data.

### swinublock.py
This module is responsible for defining the SwinU block, a specific type of building block used in constructing a neural network architecture. The SwinU block is a variation of the U-Net architecture that incorporates the Swin Transformer. These blocks are combined in various ways to form the complete model architecture.

### swinunet.py
This module defines the full SwinU-Net model architecture. SwinU-Net is a variant of the U-Net model (commonly used for segmentation tasks) that incorporates Swin Transformer components (a transformer model specifically designed for computer vision tasks). This module defines the structure of this model and includes the implementation of the forward pass, backward pass, and any other model-specific methods.

### tissuenetdata.py
This module deals with loading, preprocessing, and augmenting data for a task related to tissue analysis or classification in medical imaging. It includes functionality for reading data files, normalizing or otherwise preprocessing the data, and splitting it into training, validation, and test sets. This module might also include utilities for visualizing the data or the results of the model.

### Util Desctiptions

Each Python file in this folder contains links to GitHub with detailed descriptions.

## Execution

To execute this script, navigate to the directory containing the script and use the following command:

python3 -m topo_loss_seg.main

Make sure to satisfy all the dependencies for the project to run this script successfully. The TensorBoard logs can be viewed by pointing TensorBoard to the output directory specified in the script.

