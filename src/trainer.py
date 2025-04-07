import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, criterion, optimizer, num_epochs, num_loss_print, device):
    model.train()
    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % num_loss_print == (num_loss_print - 1):
                print(f"Epoch: {epoch + 1} / {num_epochs}, iter: {i + 1} / {num_batches}, loss: {running_loss / num_loss_print:.6f}]")
                running_loss = 0.0
    