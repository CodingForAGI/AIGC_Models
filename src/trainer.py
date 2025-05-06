import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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
    model, train_loader, criterion, optimizer, num_epochs, save_interval, save_dir, device
):
    model.train()
    num_batches = len(train_loader)
    min_loss = float('inf')
    train_loss = MeanAccumulator()

    for epoch in range(num_epochs):
        train_loss.reset()
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
            batch_iter.set_postfix({"loss": f"{avg_loss:.6f}"})
        print(f"Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_batches}, loss: {avg_loss:.6f}")
        batch_iter.close()

        # save model every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        # save model with the lowest loss
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, best_model_path)
            print(f"Best model updated and saved to {best_model_path}")