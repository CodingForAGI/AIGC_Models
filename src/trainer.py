import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from src.metric import Metric
from src.utils import PROJECT_CFG, get_log_file_path, relative_to_absolute_path, save_training_status
from torch.utils.tensorboard import SummaryWriter


def get_logger(task_name):
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(get_log_file_path(task_name)),  # write log to file
                logging.StreamHandler(),  # print log to console
            ],
        )
    logger = logging.getLogger(__name__)
    return logger


class MeanAccumulator:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    def mean(self):
        return self.sum / self.count if self.count > 0 else 0

    def reset(self):
        self.sum = 0.0
        self.count = 0


def train(
    task_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    metric: Metric,
    num_epochs: int,
    save_interval: int,
    save_dir: str,
    save_by_metric_max_value: bool,
    resume_checkpoint: str,
    device: torch.device,
):
    # logger configuration
    logger = get_logger(task_name=task_name)
    writer = SummaryWriter(log_dir=PROJECT_CFG["log_root"])

    num_batches = len(train_loader)
    train_loss = MeanAccumulator()
    if test_loader is not None:
        test_loss = MeanAccumulator()
        test_avg_loss = 0

    if save_by_metric_max_value:
        save_metric = float("-inf")   # save_metric is the best metric value, initialize save_metric to negative infinity
    else:
        save_metric = float("inf")    # initialize save_metric to positive infinity

    start_epoch = 0   # start from epoch 0 by default
    if resume_checkpoint:
        resume_checkpoint = relative_to_absolute_path(resume_checkpoint)
        if os.path.isfile(resume_checkpoint):
            logger.info(f"Loading checkpoint {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            del checkpoint
            logger.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
        else:
            logger.error(f"No checkpoint found at {resume_checkpoint}")
            raise FileNotFoundError(f"No checkpoint found at {resume_checkpoint}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss.reset()
        metric.reset()
        batch_iter = tqdm(
            enumerate(train_loader, 0),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=num_batches,
            unit="batch",
            leave=True,
        )
        for i, data in batch_iter:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))
            avg_loss = train_loss.mean()

            metric.update(outputs, labels)
            metric.compute()

            batch_iter.set_postfix(
                {
                    "loss": f"{avg_loss:.6f}",
                    **metric.to_string(key_value_fromat=True),  # Assuming to_string returns a dictionary
                }
            )

        writer.add_scalar("Train/Loss", avg_loss, epoch)
        for k, v in metric.to_string(key_value_fromat=True).items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        logger.info(
            f"[TRAIN] Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_batches}, train_loss: {avg_loss:.6f}, {metric.to_string()}"
        )
        batch_iter.close()

        # evaluate the model on validation set
        if test_loader is not None:
            model.eval()
            test_loss.reset()
            metric.reset()
            num_test_batches = len(test_loader)
            test_batch_iter = tqdm(
                enumerate(test_loader, 0),
                desc=f"Validation Epoch {epoch + 1}/{num_epochs}",
                total=num_test_batches,
                unit="batch",
                leave=True,
            )
            with torch.no_grad():
                for i, data in test_batch_iter:
                    test_inputs, test_labels = data
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

                    test_outputs = model(test_inputs)
                    test_calc_loss = criterion(test_outputs, test_labels)
                    test_loss.update(test_calc_loss.item(), test_inputs.size(0))
                    test_avg_loss = test_loss.mean()

                    metric.update(test_outputs, test_labels)
                    metric.compute()

                    # update progress bar with test loss and metric
                    test_batch_iter.set_postfix(
                        {
                            "test_loss": f"{test_avg_loss:.6f}",
                            **metric.to_string(key_value_fromat=True),  # Assuming to_string returns a dictionary
                        }
                    )


            test_batch_iter.close()
            logger.info(
                f"[VALIDATE] Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_test_batches}, test_loss: {test_avg_loss:.6f}, {metric.to_string()}"
            )
            for k, v in metric.to_string(key_value_fromat=True).items():
                writer.add_scalar(f"Test/{k}", v, epoch)

        # save model every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_training_status(epoch + 1, model, optimizer, avg_loss, checkpoint_path, save_metric)
            print(f"Model saved to {checkpoint_path}")

        # save model with best metric value
        if save_by_metric_max_value:
            # when metric is higher, save the model
            if metric.save_model_metric() > save_metric:
                save_metric = metric.save_model_metric()
                best_model_path = os.path.join(save_dir, "best_model.pth")
                save_training_status(epoch + 1, model, optimizer, avg_loss, best_model_path, save_metric)
                print(f"Best model updated and saved to {best_model_path} when metric is {save_metric}")
        else:
            # when metric is lower, save the model
            if metric.save_model_metric() < save_metric:
                save_metric = metric.save_model_metric()
                best_model_path = os.path.join(save_dir, "best_model.pth")
                save_training_status(epoch + 1, model, optimizer, avg_loss, best_model_path, save_metric)
                print(f"Best model updated and saved to {best_model_path} when metric is {save_metric}")
    

    writer.close()


def evaluate(test_loader: DataLoader, model: nn.Module, metric: Metric, device: torch.device):
    model.eval()
    metric.reset()
    num_test_batches = len(test_loader)
    test_batch_iter = tqdm(
        enumerate(test_loader, 0),
        desc="Evaluate",
        total=num_test_batches,
        unit="batch",
        leave=True,
    )
    with torch.no_grad():
        for i, data in test_batch_iter:
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)

            metric.update(test_outputs, test_labels)
            metric.compute()

            # update progress bar with test loss and metric
            test_batch_iter.set_postfix(
                {
                    **metric.to_string(key_value_fromat=True),  # Assuming to_string returns a dictionary
                }
            )
    test_batch_iter.close()
    print(f"[EVALUATE] iter: {i + 1} / {num_test_batches}, {metric.to_string()}")
