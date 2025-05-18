import torch


def get_optimizer(optimizer_name, params, lr, momentum=0.9):
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}.")
    return optimizer