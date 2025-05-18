import torch

from experiments.base_cfg import AlexNetCfg, ResNetCfg
from src.data import create_image_classification_dataloader
from src.metric import ImageClassificationMetric
from src.models.cnn_models import AlexNet
from src.models.cnn_models.resnet import get_resnet
from src.optimizer import get_optimizer
from src.trainer import evaluate, train
from src.utils import get_device, PROJECT_CFG, load_weights_from_training_status


def create_image_classification_model(model_name, cmdline_cfg):
    if model_name.lower() == "alexnet":
        cfg = AlexNetCfg().__call__(cmdline_cfg=cmdline_cfg)
        model = AlexNet(num_classes=cfg.num_classes)
        complete_model_name = "alexnet"
    elif model_name.lower() == "resnet":
        cfg = ResNetCfg().__call__(cmdline_cfg=cmdline_cfg)
        model, complete_model_name = get_resnet(scale=cfg.scale, num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}.")
    return model, cfg, complete_model_name


def image_classification_train_pipeline(args):
    device = get_device()

    # create model
    model, cfg, model_name = create_image_classification_model(model_name=args.nn, cmdline_cfg=args)
    model.to(device)

    # load data
    train_dataloader = create_image_classification_dataloader(
        dataset_name=cfg.dataset_name, batch_size=cfg.batch_size, is_train=True
    )
    test_dataloader = create_image_classification_dataloader(
        dataset_name=cfg.dataset_name, batch_size=cfg.batch_size, is_train=False
    )

    image_classification_metric = ImageClassificationMetric(
        num_classes=cfg.num_classes, eval_metric="accuracy", device=device
    )

    task_name = f"{model_name}_{cfg.dataset_name}_image_classification"

    if args.mode == "train":
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(optimizer_name=cfg.optimizer, params=model.parameters(), lr=cfg.lr)
        save_model_path = PROJECT_CFG["model_save_root"]

        train(
            task_name=task_name,
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
