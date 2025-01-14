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

    val_loader = DataLoader(
        validation_set,
        batch_size=options.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=options.num_workers,
        generator=generator,
    )

    # Simple BCE loss without weights for binary classification
    criterion = nn.BCELoss().to(model.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.learn_rate, weight_decay=options.weight_decay
    )

    val_metric = model.get_new_metric()
    train_metric = model.get_new_metric()

    for epoch in range(options.num_epochs):
        # Training phase
        model.train(True)
        train_metric.reset()

        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(model.device), y.to(model.device)

            y_pred = model.forward(X)
            loss = criterion.forward(y_pred.squeeze(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred_classes = (y_pred.squeeze() >= 0.5).long()
                train_metric.update(y_pred_classes, y)

        # Validation phase
        model.train(False)
        val_metric.reset()

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(model.device), y.to(model.device)
                y_pred = model.forward(X)
                y_pred_classes = (y_pred.squeeze() >= 0.5).long()
                val_metric.update(y_pred_classes, y)

        yield epoch, train_metric.compute(), val_metric.compute()

    model.train(False)


def predict(model: Model, vectorizer: TextVectorizer, text: str):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.vectorize(text)
        prediction = model.forward(vector.unsqueeze(0))
        return (prediction.squeeze() >= 0.5).long().item()
