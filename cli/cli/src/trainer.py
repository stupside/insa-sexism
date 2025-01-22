from .model import Model

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
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

    lr_mode: str
    lr_factor: float
    lr_patience: int

    def __init__(
        self,
        **kwargs,
    ):
        self.__dict__.update(kwargs)


def split(options: TrainOptions, dataset: TrainDataSet):
    generator = torch.Generator().manual_seed(options.seed)

    trainset_size = int(options.train_val_split * len(dataset))

    train_set, validation_set = random_split(
        lengths=[trainset_size, len(dataset) - trainset_size],
        dataset=dataset,
        generator=generator,
    )

    return train_set, validation_set


def fit(model: Model, options: TrainOptions, subset: Subset):

    loader = DataLoader(
        subset,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=model.device.type == "cpu",
        persistent_workers=options.num_workers > 0,
        num_workers=options.num_workers,
        prefetch_factor=options.num_workers if options.num_workers > 0 else None,
    )

    # Use weighted BCE loss
    criterion = nn.BCELoss()

    # Adam optimizer with weight decay for regularization and learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.learn_rate, weight_decay=options.weight_decay
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=options.lr_mode,
        factor=options.lr_factor,
        patience=options.lr_patience,
    )

    # Early stopping setup
    best_loss = float("inf")

    patience = options.lr_patience
    patience_counter = 0

    train_metrics = model.get_new_metric()

    # Set model to training mode
    model.train(True)

    X: torch.Tensor
    y: torch.Tensor

    for epoch in range(options.num_epochs):
        train_metrics.reset()

        cum_loss: torch.Tensor = torch.tensor(0.0, device=model.device)

        for X, y in loader:

            X, y = X.to(model.device), y.to(model.device)

            # Forward pass
            y_pred = model.forward(X)
            loss = criterion.forward(y_pred.squeeze(), y.float())

            # Backward pass
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # Update loss
            cum_loss += loss

            with torch.no_grad():
                y_pred_classes = (y_pred.squeeze() >= 0.5).long()
                train_metrics.update(y_pred_classes, y)

        cum_loss = cum_loss / len(loader)

        # Update learning rate based on loss
        scheduler.step(cum_loss)

        # Early stopping check
        if cum_loss < best_loss:
            best_loss = cum_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        yield epoch, train_metrics.compute(), cum_loss


def validate(model: Model, options: TrainOptions, subset: Subset):
    model.train(False)

    generator = torch.Generator().manual_seed(options.seed)

    loader = DataLoader(
        subset,
        batch_size=options.batch_size,
        shuffle=False,
        pin_memory=model.device.type == "cpu",
        num_workers=options.num_workers,
        generator=generator,
        persistent_workers=options.num_workers > 0,
        prefetch_factor=options.num_workers if options.num_workers > 0 else None,
    )

    metrics = model.get_new_metric()

    X: torch.Tensor
    y: torch.Tensor

    for X, y in loader:
        X, y = X.to(model.device), y.to(model.device)

        # Forward pass
        y_pred = model.forward(X)
        y_pred_classes = (y_pred.squeeze() >= 0.5).long()

        # Update validation metric
        metrics.update(y_pred_classes, y)

        yield metrics.compute()


def predict(model: Model, vectorizer: TextVectorizer, text: str):
    model.eval()

    vector = vectorizer.vectorize(text).to(model.device)

    prediction = model.forward(vector)

    classes = prediction.squeeze() >= 0.5

    return classes.long()
