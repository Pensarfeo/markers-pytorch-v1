import torch
import torch.nn as nn

def build(model, lr=0.001, weightDecay=1e-5, momentum=0.9):
    """
    Builds an optimizer with the specified parameters and parameter groups.

    Args:
        model (nn.Module): model to optimize
        name (str): name of the optimizer to use
        lr (float): learning rate
        momentum (float): momentum
        decay (float): weight decay

    Returns:
        optimizer (torch.optim.Optimizer): the built optimizer
    """
    g = [], [], []  # optimizer parameter groups
    
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
    
    # if name == 'Adam':
    # optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    # elif name == 'AdamW':
    # optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    # elif name == 'RMSProp':
    optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum,  foreach=True)
    # elif name == 'SGD':
    # optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    # else:
    #     raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': weightDecay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer