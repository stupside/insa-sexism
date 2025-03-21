import csv
import typer

from typing_extensions import Annotated

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
    min_document_frequency: int = 2,
):

    from ast import literal_eval
    from .cmd.train import TrainSet, TrainData, run

    trainset = TrainSet()
    reader = csv.DictReader(file)

    from rich.progress import track

    for row in track(reader, description="Loading CSV file"):
        # Convert string representations of arrays to actual arrays
        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # Keep original string if parsing fails

        trainset.add(TrainData(**row))

    run(
        trainset,
        seed,
        top_k,
        token_mode,
        ngram_range,
        min_document_frequency,
    )
