import torch
from util.pos_embed import interpolate_pos_embed

def load_checkpoint(net, CHECKPOINT_PATH):
    print("Loading pre-trained checkpoint from: %s" % CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = net.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(net, checkpoint_model)

    # load pre-trained model
    msg = net.load_state_dict(checkpoint_model, strict=False)

    print(msg)

    return net