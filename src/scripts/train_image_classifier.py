import torch
from src.data import create_image_classification_dataloader
from src.models.cnn_models import AlexNet
from src.trainer import train
from src.utils import get_device


if __name__ == "__main__":
    device = get_device()
    model = AlexNet(num_classes=10).to(device)
    train_dataloader = create_image_classification_dataloader(
        dataset_name="cifar10", batch_size=64, is_train=True
    )
    test_dataloader = create_image_classification_dataloader(
        dataset_name="cifar10", batch_size=64, is_train=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 50

    train(
        model=model,
        train_loader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        num_loss_print=50,
        device=device,
    )