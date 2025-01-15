import torch
import torch.nn as nn

from .metric import Metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelOptions:
    output_dim: int
    layer_dims: list[int]
    dropout_rate: float
    embedding_dim: int

    def __init__(
        self,
        **kwargs,
    ):
        self.__dict__.update(kwargs)


class Model(nn.Module):

    device: torch.device

    def __init__(self, device: torch.device, vocab_size: int, options: ModelOptions):
        super(Model, self).__init__()

        self.device = device

        # Add embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=options.embedding_dim
        )

        self.seq = nn.Sequential()

        # Modify first layer to use embedding dimension
        self.seq.add_module("norm1d", nn.BatchNorm1d(options.embedding_dim))

        prev_dim = options.embedding_dim

        # Add dynamic number of layers
        for i in range(len(options.layer_dims)):
            current_dim = options.layer_dims[i]

            # Add dense layer with ReLU activation
            self.seq.add_module(f"layer{i}_linear", nn.Linear(prev_dim, current_dim))
            self.seq.add_module(f"layer{i}_relu", nn.ReLU())
            # Add batch normalization
            self.seq.add_module(f"layer{i}_norm1d", nn.BatchNorm1d(current_dim))
            # Add dropout
            self.seq.add_module(f"layer{i}_dropout", nn.Dropout(options.dropout_rate))

            prev_dim = current_dim

        # Add output layer
        self.seq.add_module("linear", nn.Linear(prev_dim, options.output_dim))

        # Add sigmoid activation for binary classification
        if options.output_dim == 1:
            self.seq.add_module("sigmoid", nn.Sigmoid())
        else:
            self.seq.add_module("softmax", nn.Softmax(dim=-options.output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run embedding layer
        x = self.embedding.forward(x)

        # Average embeddings
        x = torch.mean(x, dim=1)

        return self.seq.forward(x)

    @staticmethod
    def get_new_metric():
        return Metric(device=device)

    @staticmethod
    def get(vocab_size: int, options: ModelOptions) -> "Model":

        classifier = Model(device=device, vocab_size=vocab_size, options=options)

        return classifier
