

class AlexNetCfg:
    def __init__(self, args=None):
        # default config params
        self.num_classes = 1000
        self.batch_size = 128
        self.epochs = 100
        self.lr = 0.001

        # dataset
        self.dataset_name = "cifar10"


        self._apply_cmdline_config(args)

    def _apply_cmdline_config(cmdline_cfg):
        pass
