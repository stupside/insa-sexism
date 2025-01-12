from .model import Model

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from cli.src.sets.train import TrainDataSet
from cli.src.vectorizer import TextVectorizer


class TrainOptions:

    num_epochs: int
    batch_size: int
    learn_rate: float
    num_workers: int
    weight_decay: float
    cross_entropy_weight: tuple[float, float]

    def __init__(
        self,
        num_epochs: int,
        num_workers: int,
        batch_size: int,
        learn_rate: float,
        weight_decay: float,
        cross_entropy_weight: tuple[float, float],
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.cross_entropy_weight = cross_entropy_weight


def fit(model: Model, options: TrainOptions, dataset: TrainDataSet):

    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=options.num_workers,
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(options.cross_entropy_weight).to(model.device)
    ).to(model.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=options.learn_rate, weight_decay=options.weight_decay
    )

    for epoch in range(options.num_epochs):
        model.train(True)

        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(model.device), y.to(model.device)

            # Forward pass
            y_pred = model.forward(X)

            loss = criterion.forward(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            with torch.no_grad():
                # Convert predictions to class indices for metric calculation
                y_pred_classes = torch.argmax(y_pred, dim=1)
                model.metric.update(y_pred_classes, y)

        yield epoch, model.metric.compute()

    model.metric.reset()


def predict(model: Model, vectorizer: TextVectorizer, text: str):
    model.eval()

    vector = vectorizer.vectorize(text)
    prediction = model.forward(vector.unsqueeze(0))

    prediction = torch.argmax(prediction, dim=1)

    return prediction
