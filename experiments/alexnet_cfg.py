import os

from src.utils import get_format_time, get_repo_root, load_yaml_config, save_to_yaml


class AlexNetCfg:
    def __init__(self):
        # default config params
        self.num_classes = 1000
        self.batch_size = 128
        self.epochs = 100
        self.lr = 0.001

        # dataset
        self.dataset_name = "cifar10"

        self.repo_root = get_repo_root()
    
    def __call__(self, cmdline_cfg=None):
        if cmdline_cfg.cfg is not None:
            yaml_cfg = load_yaml_config(os.path.join(self.repo_root, "experiments", cmdline_cfg.cfg))
            self._apply_yaml_config(yaml_cfg)
        self._apply_cmdline_config(cmdline_cfg)
        self._write_cfg_to_file()
        return self

    def _apply_cmdline_config(self, cmdline_cfg):
        for key, value in vars(cmdline_cfg).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    

    def _apply_yaml_config(self, yaml_cfg):
        for key, value in yaml_cfg.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def _write_cfg_to_file(self):
        current_time = get_format_time()
        experiment_cfg_path = os.path.join(self.repo_root, "experiments", f"alexnet_cfg_{current_time}.yaml")
        if 'repo_root' in self.__dict__:
            # delete repo_root key-value pair
            del self.__dict__['repo_root']
        save_to_yaml(self.__dict__, experiment_cfg_path)
        print(f"Successfully saved config file to: {experiment_cfg_path}.")