import torch
import torch.nn as nn
import os

from .metric import Metric


device: torch.device

# if torch.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class EmbeddingOptions:
    num_epochs: int
    batch_size: int
    learn_rate: float
    weight_decay: float
    embedding_dim: int
    checkpoint_path: str

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ModelOptions:
    output_dim: int
    layer_dims: list[int]
    dropout_rate: float
    embedding_dim: int

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class EmbeddingModel(nn.Module):
    @staticmethod
    def load(options: EmbeddingOptions, vocab_size: int) -> "EmbeddingModel":
        model = EmbeddingModel(vocab_size, options.embedding_dim)
        if os.path.exists(options.checkpoint_path):
            model.load_state_dict(
                torch.load(options.checkpoint_path, weights_only=True)
            )
        return model

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def freeze(self):
        """Freeze embedding parameters"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze embedding parameters"""
        for param in self.parameters():
            param.requires_grad = True


class ClassifierModel(nn.Module):

    device: torch.device
    options: ModelOptions

    def __init__(self, embedding_model: EmbeddingModel, options: ModelOptions):
        super(ClassifierModel, self).__init__()

        self.device = device
        self.embedding_model = embedding_model
        self.options = options

        layers = []
        prev_dim = options.embedding_dim

        for dim in options.layer_dims:
            layers.extend(
                [nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(options.dropout_rate)]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, options.output_dim))

        self.layers = nn.Sequential(*layers)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_model(x)
        x = torch.mean(x, dim=1)
        return self.layers(x)

    @staticmethod
    def get_new_metric():
        return Metric(device=device)

    @staticmethod
    def get(
        vocab_size: int,
        options: ModelOptions,
        embedding_model: EmbeddingModel,
    ):
        if embedding_model is None:
            embedding_model = EmbeddingModel(vocab_size, options.embedding_dim)
        return ClassifierModel(embedding_model, options)
