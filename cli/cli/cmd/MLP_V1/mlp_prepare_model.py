# Tensorflow related imports
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras import models

# Custom imports
from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat
from cli.cmd.MLP_V1.mlp_preprocess_data import MLP_PREPOCESS_DATA
from cli.cmd.MLP_V1.wrapper import MLP_MODEL_PARAMS_WRAPPER, MLP_PREPOCESS_PARAM_WRAPPER


class MLP_MODEL:
    """
    This class is used to prepare the model for the MLP
    """

    num_classes: int = 2

    def __init__(
        self,
        params: MLP_MODEL_PARAMS_WRAPPER,
    ):
        self.params = params

    def __get_last_layer_units_and_activation(self):

        if self.num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = self.num_classes
        return units, activation

    def __mlp_model(self):

        op_units, op_activation = self.__get_last_layer_units_and_activation()
        model = models.Sequential()
        model.add(
            Dropout(rate=self.params.dropout_rate, input_shape=self.params.input_shape)
        )

        for _ in range(self.params.layers - 1):
            model.add(Dense(units=self.params.units, activation="relu"))
            model.add(Dropout(rate=self.params.dropout_rate))

        model.add(Dense(units=op_units, activation=op_activation))
        return model

    def get_model(
        self,
        training_data: list[TrainingDataFormat],
        validation_data: list[ValidationDataFormat],
        vectorizer_param: MLP_PREPOCESS_PARAM_WRAPPER,
        top_k: int,
    ):
        # Preprocess the data
        data_preprocessor = MLP_PREPOCESS_DATA(
            training_data, validation_data, vectorizer_param, top_k
        )
        data_preprocessor.run()

        # Prepare the model
        return self.__mlp_model()


if __name__ == "main":
    print("You have selected the MLP_V1 model")
