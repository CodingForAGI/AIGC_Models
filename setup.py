import argparse
from src.pipelines import image_classification_train_pipeline


def parse_cmdline_param():
    parser = argparse.ArgumentParser(description="setting training parameters")
    parser.add_argument("--mode", type=str, help="mode of task(such as train, eval)")
    parser.add_argument("--task", type=str, help="task name")
    parser.add_argument("--nn", type=str, help="neural network name")
    parser.add_argument("--scale", type=str, help="scale of model")
    parser.add_argument("--model_path", type=str, help="path of model")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--epochs", type=int, help="num of epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--save_interval", type=int, help="save model interval")
    parser.add_argument("--num_classes", type=int, help="number of classes")
    parser.add_argument("--resume_ckpt", type=str, help="path of resume checkpoint")
    parser.add_argument("--cfg", type=str, help="yaml config file name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cmdline_param()
    task_name = args.task
    if task_name.lower() == "img_cls":
        image_classification_train_pipeline(args=args)
