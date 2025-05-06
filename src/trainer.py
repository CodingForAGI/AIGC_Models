import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from src.metric import Metric
from src.utils import get_log_file_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(get_log_file_path()),  # write log to file
        logging.StreamHandler(),  # print log to console
    ],
)
logger = logging.getLogger(__name__)


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
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    metric: Metric,
    num_epochs: int,
    save_interval: int,
    save_dir: str,
    device: torch.device,
):
    model.train()
    num_batches = len(train_loader)
    min_loss = float("inf")
    train_loss = MeanAccumulator()

    for epoch in range(num_epochs):
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
            f"Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_batches}, loss: {avg_loss:.6f}, {metric.to_string()}"
        )
        batch_iter.close()

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
