
def get_layer_id_for_resnext(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    # print(name)
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif 'patch_embed' in name:
        return 0
    elif 'layer' in name:
        nmlst = name.split('.')
        return int(nmlst[1][-1])*5 + int(nmlst[2])
    else:
        return num_layers
