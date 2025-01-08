import csv
import typer

from typing_extensions import Annotated
from ast import literal_eval
from .cmd.train import TrainSet, TrainData, run

from rich.progress import track

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
    file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    seed: int = 123,
    top_k: int = 20000,
    token_mode: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_document_frequency: float = 0.01,
):

    # run data_analysis
    trainset = TrainSet()
    run(
        trainset,
        seed,
        top_k,
        token_mode,
        ngram_range,
        min_document_frequency,
    )


def read_csv_file(file: typer.FileText, moduleName: str, className: str):
    # Import the module dynamically
    module = __import__(moduleName)

    # Fetch the class and instantiate it
    ClassToInstantiate = getattr(module, className)

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

        tmp_list.append(ClassToInstantiate(**row))

    return tmp_list
