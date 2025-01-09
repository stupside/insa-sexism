import csv
import typer

from typing_extensions import Annotated
from ast import literal_eval

from rich.progress import track

from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat

app = typer.Typer(
    help="This is a CLI tool detect sexism in text.", no_args_is_help=True
)


@app.command()
def ping():
    typer.echo("Pong")


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def train(
    file_training_data: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    file_testing_data: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    model: str = "MLP_V1",
    seed: int = 123,
    top_k: int = 20000,
    token_mode: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_document_frequency: float = 0.01,
):
    # load training data
    # train_set = read_csv_file(file_training_data, "trainning")
    # load validation data
    # validation_set = read_csv_file(file_testing_data, "validation")

    match model:
        case "MLP_V1":
            print("You have selected the MLP_V1 model")

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
