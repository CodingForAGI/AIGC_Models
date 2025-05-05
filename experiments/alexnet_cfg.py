import os

from src.utils import get_format_time, get_repo_root, save_to_yaml


class AlexNetCfg:
    def __init__(self):
        # default config params
        self.num_classes = 1000
        self.batch_size = 128
        self.epochs = 100
        self.lr = 0.001

        # dataset
        self.dataset_name = "cifar10"
    
    def __call__(self, cmdline_cfg=None, yaml_cfg=None):
        if yaml_cfg is not None:
            self._apply_yaml_config(yaml_cfg)
        if cmdline_cfg is not None:
            self._apply_cmdline_config(cmdline_cfg)
        self._write_cfg_to_file()
        return self

    def _apply_cmdline_config(self, cmdline_cfg):
        self.batch_size = cmdline_cfg.batch_size
        self.epochs = cmdline_cfg.epoch
        self.lr = cmdline_cfg.lr
    

    def _apply_yaml_config(self, yaml_cfg):
        pass

    def _write_cfg_to_file(self):
        repo_root = get_repo_root()
        current_time = get_format_time()
        experiment_cfg_path = os.path.join(repo_root, "experiments", f"alexnet_cfg_{current_time}.yaml")
        save_to_yaml(self.__dict__, experiment_cfg_path)
        print(f"Successfully save config file to: {experiment_cfg_path}.")