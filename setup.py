
import argparse
from src.scripts import train_alexnet_on_cifar10


def parse_cmdline_param():
    parser = argparse.ArgumentParser(description='setting training parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epoch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_alexnet_on_cifar10(args=parse_cmdline_param())