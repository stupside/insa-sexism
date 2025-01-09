# Tensorflow related imports
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import models

# Custom imports
from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat


class MLP_MODEL:
    """
    This class is used to prepare the model for the MLP
    """

    def __init__(self):
        pass

    def _get_last_layer_units_and_activation(num_classes):
        """Gets the # units and activation function for the last network layer.

        # Arguments
            num_classes: int, number of classes.

        # Returns
            units, activation values.
        """
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        return units, activation

    def mlp_model(
        self,
        layers: int,
        units: int,
        dropout_rate: float,
        input_shape: tuple,
        num_classes: int,
    ):
        """Creates an instance of a multi-layer perceptron model.

        # Arguments
            layers: int, number of `Dense` layers in the model.
            units: int, output dimension of the layers.
            dropout_rate: float, percentage of input to drop at Dropout layers.
            input_shape: tuple, shape of input to the model.
            num_classes: int, number of output classes.

        # Returns
            An MLP model instance.
        """
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)
        model = models.Sequential()
        model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

        for _ in range(layers - 1):
            model.add(Dense(units=units, activation="relu"))
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(units=op_units, activation=op_activation))
        return model


def run(
    training_data: list[TrainingDataFormat],
    validation_data: list[ValidationDataFormat],
    seed: int,
    top_k: int,
    token_mode: str,
    ngram_range: tuple[int, int],
    min_document_frequency: int,
):
    print("hello")
