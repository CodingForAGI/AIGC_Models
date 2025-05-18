import os

from experiments.base_cfg import BaseCfg
from src.utils import get_format_time, get_repo_root, load_yaml_config, save_to_yaml


class AlexNetCfg(BaseCfg):
    def __init__(self, cfg_name="alexnet_cfg"):
        super().__init__(cfg_name)
        # default config params
        self.num_classes = 10
        self.batch_size = 128
        self.epochs = 100
        self.lr = 0.001

        # dataset
        self.dataset_name = "cifar10"

        # save model
        self.save_interval = 10