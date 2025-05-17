import torch
import os

from experiments.alexnet_cfg import AlexNetCfg
from src.data import create_image_classification_dataloader
from src.metric import ImageClassificationMetric
from src.models.cnn_models import AlexNet
from src.trainer import evaluate, train
from src.utils import get_device, PROJECT_CFG, load_weights_from_training_status


def train_alexnet_on_cifar10(args):
    cfg = AlexNetCfg().__call__(cmdline_cfg=args)
    device = get_device()
    model = AlexNet(num_classes=cfg.num_classes).to(device)
    train_dataloader = create_image_classification_dataloader(
        dataset_name=cfg.dataset_name, batch_size=cfg.batch_size, is_train=True
    )
    test_dataloader = create_image_classification_dataloader(
        dataset_name=cfg.dataset_name, batch_size=cfg.batch_size, is_train=False
    )

    image_classification_metric = ImageClassificationMetric(num_classes=cfg.num_classes, eval_metric="accuracy", device=device)

    if args.mode == "train":
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
        save_model_path = PROJECT_CFG["model_save_root"]

        train(
            task_name="alexnet_cifar10",
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            metric=image_classification_metric,
            num_epochs=cfg.epochs,
            save_interval=cfg.save_interval,
            save_dir=save_model_path,
            save_by_metric_max_value=True,
            device=device,
        )
    elif args.mode == "eval":
        # load weights from model_path
        load_weights_from_training_status(model=model, weights_path=args.model_path, device=device)
        evaluate(test_loader=test_dataloader, model=model, metric=image_classification_metric, device=device)
    else:
        raise ValueError(f"Invalid mode name: {args.mode}. Please choose from 'train' and 'eval'.")
