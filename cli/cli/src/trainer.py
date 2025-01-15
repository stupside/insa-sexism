from .model import Model

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from cli.src.sets.train import TrainDataSet
from cli.src.vectorizer import TextVectorizer


class TrainOptions:

    seed: int
    num_epochs: int
    batch_size: int
    learn_rate: float
    num_workers: int
    weight_decay: float
    train_val_split: float

    def __init__(
        self,
        **kwargs,
    ):
        self.__dict__.update(kwargs)


def fit(model: Model, options: TrainOptions, dataset: TrainDataSet):

    torch.random.manual_seed(options.seed)

    generator = torch.Generator().manual_seed(options.seed)

    trainset_size = int(options.train_val_split * len(dataset))

    train_set, validation_set = random_split(
        lengths=[trainset_size, len(dataset) - trainset_size],
        dataset=dataset,
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=options.num_workers,
    )

    # Simple BCE loss without weights for binary classification
    criterion = nn.BCELoss()

    # Adam optimizer with weight decay for regularization and learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.learn_rate, weight_decay=options.weight_decay
    )

    train_metrics = model.get_new_metric()

    # Set model to training mode
    model.train(True)

    X: torch.Tensor
    y: torch.Tensor

    print("Training phase")

    for epoch in range(options.num_epochs):
        # Training phase
        train_metrics.reset()

        for X, y in train_loader:

            X, y = X.to(model.device), y.to(model.device)

            # Forward pass
            y_pred = model.forward(X)
            loss = criterion.forward(y_pred.squeeze(), y.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred_classes = (y_pred.squeeze() >= 0.5).long()
                train_metrics.update(y_pred_classes, y)

        print(f"Epoch {epoch + 1}/{options.num_epochs} - {train_metrics.compute()}")

    # Set model to evaluation mode
    model.train(False)

    val_loader = DataLoader(
        validation_set,
        batch_size=options.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=options.num_workers,
        generator=generator,
    )

    val_metrics = model.get_new_metric()

    print("Validation phase")

    for X, y in val_loader:
        X, y = X.to(model.device), y.to(model.device)

        # Forward pass
        y_pred = model.forward(X)
        y_pred_classes = (y_pred.squeeze() >= 0.5).long()

        # Update validation metric
        val_metrics.update(y_pred_classes, y)

    return train_metrics.compute(), val_metrics.compute()


def predict(model: Model, vectorizer: TextVectorizer, text: str):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.vectorize(text)
        prediction = model.forward(vector.unsqueeze(0))
        return (prediction.squeeze() >= 0.5).long().item()
