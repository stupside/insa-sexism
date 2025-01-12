import torch
import torch.nn as nn

from .metric import Metric


class ModelOptions:

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    dropout_rate: float

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate


class Model(nn.Module):

    device: torch.device
    metric: Metric

    def __init__(self, metric: Metric, device: torch.device, options: ModelOptions):
        super(Model, self).__init__()

        self.device = device
        self.metric = metric

        self.seq = nn.Sequential()

        # Add normalization layer
        self.seq.add_module("norm1d", nn.BatchNorm1d(options.input_dim))

        # Add dynamic number of layers
        for i in range(options.num_layers):
            # Get previous layer dimension for connecting layers
            prev_dim = options.hidden_dim if i > 0 else options.input_dim

            # Add dense layer with ReLU activation
            self.seq.add_module(
                f"layer{i}_linear", nn.Linear(prev_dim, options.hidden_dim)
            )
            self.seq.add_module(f"layer{i}_relu", nn.ReLU())
            # Add batch normalization
            self.seq.add_module(f"layer{i}_norm1d", nn.BatchNorm1d(options.hidden_dim))
            # Add dropout
            self.seq.add_module(f"layer{i}_dropout", nn.Dropout(options.dropout_rate))

        # Add output layer
        self.seq.add_module("linear", nn.Linear(options.hidden_dim, options.output_dim))

        # Add sigmoid activation for binary classification
        self.seq.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.seq.forward(x)

    @staticmethod
    def get(options: ModelOptions) -> "Model":

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metric = Metric(device=device)

        classifier = Model(device=device, metric=metric, options=options)

        return classifier.to(device)
