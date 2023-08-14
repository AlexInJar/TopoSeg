import torch
from util import misc

@torch.no_grad()
def evaluate(data_loader,criterion, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch['image']
        targets = batch['ind']
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss, mse_v, mmt_v = criterion(output, targets)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(mse=mse_v)
        metric_logger.update(mmt=mmt_v)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}