import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from src.metric import Metric
from src.utils import get_log_file_path


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
    device: torch.device,
):
    # logger configuration
    logger = get_logger(task_name=task_name)

    num_batches = len(train_loader)
    min_loss = float("inf")
    train_loss = MeanAccumulator()
    if test_loader is not None:
        test_loss = MeanAccumulator()

    for epoch in range(num_epochs):
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
                    **metric.to_string(
                        key_value_fromat=True
                    ),  # Assuming to_string returns a dictionary
                }
            )
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
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(
                        device
                    )

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
                            **metric.to_string(
                                key_value_fromat=True
                            ),  # Assuming to_string returns a dictionary
                        }
                    )
            test_batch_iter.close()
            logger.info(
                f"[VALIDATE] Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_test_batches}, test_loss: {test_avg_loss:.6f}, {metric.to_string()}"
            )

        # save model every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Model saved to {checkpoint_path}")

        # save model with the lowest loss
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                best_model_path,
            )
            print(f"Best model updated and saved to {best_model_path}")


def evaluate(
    test_loader: DataLoader, model: nn.Module, metric: Metric, device: torch.device
):
    model.eval()
    test_loss = MeanAccumulator()
    test_loss.reset()
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
                    **metric.to_string(
                        key_value_fromat=True
                    ),  # Assuming to_string returns a dictionary
                }
            )
    test_batch_iter.close()
    print(
        f"[EVALUATE] iter: {i + 1} / {num_test_batches}, test_loss: {test_avg_loss:.6f}, {metric.to_string()}"
    )
