import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(
    model, train_loader, criterion, optimizer, num_epochs, num_loss_print, device
):
    model.train()
    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        running_loss = 0.0
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

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            batch_iter.set_postfix({"loss": f"{avg_loss:.6f}"})
        print(f"Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_batches}, loss: {avg_loss:.6f}")
        batch_iter.close()
