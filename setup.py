
import argparse
from src.scripts import train_alexnet_on_cifar10


def parse_cmdline_param():
    parser = argparse.ArgumentParser(description='setting training parameters')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--save_interval', type=int, help='save model interval')
    parser.add_argument('--cfg', type=str, help='yaml config file name')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_alexnet_on_cifar10(args=parse_cmdline_param())