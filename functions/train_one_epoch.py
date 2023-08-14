import torch
from util import misc
from timm.data import Mixup
from functions.adjust_learning_rate import adjust_learning_rate
import sys
import math
from typing import Iterable, Optional

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #{
    metric_logger.add_meter('reconst', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('topo', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('tae', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cnct', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #}
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = 1

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print(item[1].keys())
        break


    for data_iter_step, datum in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch)

        samples, targets = datum['image'], datum['ind']
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            #{1
            outputs,latent = model(samples)
            loss, reconst_v, topo_v, tae_v, cnct_v = criterion(outputs, samples, latent, targets)
            #1}

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
         #{1
        metric_logger.update(loss=loss_value)
        metric_logger.update(reconst=reconst_v)
        metric_logger.update(topo=topo_v)
        metric_logger.update(tae=tae_v)
        metric_logger.update(cnct=cnct_v)
        #1}

        min_lr = 10.
        max_lr = 0.

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        #{1
        reconst_reduce = misc.all_reduce_mean(reconst_v)
        topo_reduce = misc.all_reduce_mean(topo_v)
        tae_reduce = misc.all_reduce_mean(tae_v)
        cnct_reduce = misc.all_reduce_mean(cnct_v)
        #1}

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}