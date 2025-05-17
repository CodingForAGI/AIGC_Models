import argparse
from src.scripts import train_alexnet_on_cifar10


def parse_cmdline_param():
    parser = argparse.ArgumentParser(description="setting training parameters")
    parser.add_argument("--mode", type=str, help="mode of task(such as train, eval)")
    parser.add_argument("--nn", type=str, help="neural network name")
    parser.add_argument("--model_path", type=str, help="path of model")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--epochs", type=int, help="num of epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--save_interval", type=int, help="save model interval")
    parser.add_argument("--num_classes", type=str, help="number of classes")
    parser.add_argument("--cfg", type=str, help="yaml config file name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cmdline_param()
    model_name = args.nn
    dataset_name = args.dataset
    if model_name.lower() == "alexnet" and dataset_name.lower() == "cifar10":
        train_alexnet_on_cifar10(args=args)
