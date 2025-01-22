from .model import EmbeddingModel, ClassifierModel, EmbeddingOptions

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
    checkpoint_path: str = "./out/model.pt"

    def __init__(self, **kwargs):
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


def load_model(model: nn.Module, options: EmbeddingOptions):
    """Load a model's weights safely"""
    model.load_state_dict(torch.load(options.checkpoint_path, weights_only=True))
    return model


def fit_embedding(
    embedding_model: EmbeddingModel, options: EmbeddingOptions, subset: Subset
):
    """Train the embedding model separately"""
    if os.path.exists(options.checkpoint_path):
        embedding_model = load_model(embedding_model, options)
        return []

    loader = DataLoader(
        subset,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=embedding_model.device.type == "cpu",
        num_workers=0,
        prefetch_factor=None,
    )

    # Change to use cosine similarity loss
    criterion = nn.CosineSimilarity(dim=-1)

    optimizer = torch.optim.Adam(
        embedding_model.parameters(),
        lr=options.learn_rate,
        weight_decay=options.weight_decay,
    )

    embedding_model.train(True)

    for epoch in range(options.num_epochs):
        cum_loss = 0.0

        for X, _ in loader:
            X = X.to(embedding_model.device)

            # Get current and next tokens for prediction
            input_ids = X[:, :-1]
            target_ids = X[:, 1:]

            # Get embeddings and normalize them
            input_embeds = embedding_model(input_ids)
            target_embeds = embedding_model(target_ids)

            # Normalize embeddings
            input_embeds = F.normalize(input_embeds, p=2, dim=-1)
            target_embeds = F.normalize(target_embeds, p=2, dim=-1)

            # Compute similarity loss (1 - cosine similarity for minimization)
            loss = (1 - criterion(input_embeds, target_embeds)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

        yield epoch, cum_loss / len(loader)


def fit(model: ClassifierModel, options: TrainOptions, subset: Subset):

    loader = DataLoader(
        subset,
        batch_size=options.batch_size,
        shuffle=True,
        pin_memory=model.device.type == "cpu",
        persistent_workers=options.num_workers > 0,
        num_workers=options.num_workers,
        prefetch_factor=options.num_workers if options.num_workers > 0 else None,
    )

    # Load previous best model if exists with weights_only=True
    if hasattr(options, "checkpoint_path") and os.path.exists(options.checkpoint_path):
        model.load_state_dict(torch.load(options.checkpoint_path, weights_only=True))

    # Initialize bias for better class balance
    if hasattr(model.layers[-2], "bias"):
        pos_weight = sum(1 for _, y in subset if y == 0) / sum(
            1 for _, y in subset if y == 1
        )
        model.layers[-2].bias.data = torch.log(torch.tensor([pos_weight]))

    # Use weighted BCE loss for class imbalance
    pos_weight = torch.tensor(
        [sum(1 for _, y in subset if y == 0) / sum(1 for _, y in subset if y == 1)]
    ).to(model.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    # Load previous best model if exists
    if hasattr(options, "checkpoint_path") and os.path.exists(options.checkpoint_path):
        model.load_state_dict(torch.load(options.checkpoint_path, weights_only=True))

    # Check if dataset is empty or has only one class
    labels = [y for _, y in subset]
    if not labels or len(set(labels)) < 2:
        return

    # Calculate class weights only if both classes exist
    pos_samples = sum(1 for _, y in subset if y == 1)
    neg_samples = sum(1 for _, y in subset if y == 0)

    if pos_samples == 0 or neg_samples == 0:
        pos_weight = torch.tensor([1.0]).to(model.device)
    else:
        pos_weight = torch.tensor([neg_samples / pos_samples]).to(model.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state_dict = None

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

        # Early stopping check and model saving
        if cum_loss < best_loss:
            best_loss = cum_loss
            patience_counter = 0
            best_state_dict = model.state_dict().copy()  # Save the best model state
            if hasattr(options, "checkpoint_path"):
                torch.save(best_state_dict, options.checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if best_state_dict is not None:
                model.load_state_dict(
                    best_state_dict
                )  # Restore best model before stopping
            break

        yield epoch, train_metrics.compute(), cum_loss

    # Ensure we're using the best model state at the end
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)


def validate(model: ClassifierModel, options: TrainOptions, subset: Subset):
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

    # Check if validation set is empty
    if len(subset) == 0:
        return

    # Initialize metrics with empty class handling
    metrics = model.get_new_metric()

    # Track unique classes in validation set
    seen_classes = set()

    X: torch.Tensor
    y: torch.Tensor

    for X, y in loader:
        X, y = X.to(model.device), y.to(model.device)
        seen_classes.update(y.unique().cpu().numpy())

        # Forward pass
        y_pred = model.forward(X)
        y_pred_classes = (y_pred.squeeze() >= 0.5).long()

        # Update validation metric with safe handling
        metrics.update(y_pred_classes, y)

    yield metrics.compute()


def predict(model: ClassifierModel, vectorizer: TextVectorizer, text: str):
    model.eval()

    vector = vectorizer.vectorize(text).to(model.device)

    prediction = model.forward(vector)

    classes = prediction.squeeze() >= 0.5

    return classes.long()
