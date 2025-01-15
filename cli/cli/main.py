import csv
import typer

from typing_extensions import Annotated
from ast import literal_eval

from rich.progress import track

from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat

# custom imports
# from cli.cmd.MLP_V1.mlp_prepare_model import MLP_MODEL
from cli.cmd.MLP_V1.wrapper import MLP_MODEL_PARAMS_WRAPPER, MLP_PREPOCESS_PARAM_WRAPPER
import cli.cmd.sepCNN.main as sepCNN

app = typer.Typer(
    help="This is a CLI tool detect sexism in text.", no_args_is_help=True
)


@app.command()
def train(
    file_training_data: Annotated[
        typer.FileText, typer.Option(encoding="UTF-8")
    ] = "../cli-data/train.csv",
    file_testing_data: Annotated[
        typer.FileText, typer.Option(encoding="UTF-8")
    ] = "../cli-data/test.csv",
    model: str = "sepCNN",
):
    # load training data
    train_set = read_csv_file(file_training_data, "trainning")
    # load validation data
    validation_set = read_csv_file(file_testing_data, "validation")

    match model:
        case "MLP_V1":
            # User message
            print("You have selected the MLP_V1 model")
            # Vars
            top_k = 20000
            mlp_model_params = MLP_MODEL_PARAMS_WRAPPER(
                num_classes=2, layes=2, units=4, dropout_rate=0.3, input_shape=(1, 5)
            )

            mlp_preprocess_params = MLP_PREPOCESS_PARAM_WRAPPER(
                dtype=float,
                min_df=1,
                analyzer="word",
                ngram_range=(1, 2),
                decode_error="strict",
            )
            # get the model
            mlp_model = MLP_MODEL(mlp_model_params)
            model = mlp_model.get_model(
                train_set, validation_set, mlp_preprocess_params, top_k
            )

        case "sepCNN":
            print("You have selected the sepCNN model")
            # test preprocess data
            # sepCNN.preprocess_data(train_set, validation_set)
            sepCNN.printHello()
            ((tweets1, labels1), (tweets2, labels2)) = sepCNN.prepare_data(train_set)
            print(len(labels2), len(tweets2))
            print(len(labels1), len(tweets1))
            print(len(labels2) + len(labels1))
            pass
        case _:
            print("You haven't selected a model yet")


def read_csv_file(file: typer.FileText, dataType: str):
    reader = csv.DictReader(file)
    tmp_list = []

    for row in track(reader, description="Loading CSV file"):
        # Convert string representations of arrays to actual arrays
        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # Keep original string if parsing fails
        if dataType == "trainning":
            instance = TrainingDataFormat(**row)
        else:
            instance = ValidationDataFormat(**row)
        tmp_list.append(instance)

    return tmp_list
