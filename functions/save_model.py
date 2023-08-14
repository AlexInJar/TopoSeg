import sys
from pathlib import Path
from functions.save_on_master import save_on_master
import re

def save_model(output_dirnm, epoch, model, model_without_ddp, optimizer, loss_scaler):
    
    output_dir = Path(output_dirnm)

    def extract_epoch_num(filename):
        match = re.search(r'checkpoint-(\d+).pth', filename)
        return int(match.group(1)) if match else 0

    # Check for existing checkpoints
    existing_checkpoints = list(output_dir.glob('checkpoint-*.pth'))
    if existing_checkpoints:
        # Sort by epoch number
        last_checkpoint_name = sorted(existing_checkpoints, key=lambda x: extract_epoch_num(x.name))[-1].name
        last_epoch_num = extract_epoch_num(last_checkpoint_name)
        epoch = last_epoch_num + 1

    epoch_name = str(epoch)
    print("[SAVED EPOCH NUMBER]->>> ", epoch)

    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict()
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=output_dirnm, tag="checkpoint-%s" % epoch_name, client_state=client_state)

    return epoch