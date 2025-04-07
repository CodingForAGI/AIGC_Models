import os
import torch

def get_repo_root():
    cur_file = os.path.abspath(__file__)
    repo_root = os.path.normpath(os.path.join(cur_file, "../"))
    return repo_root


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)