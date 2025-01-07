import csv
import time
import typer

from typing_extensions import Annotated

app = typer.Typer(help="This is a CLI tool detect sexism in text.", no_args_is_help=True)

@app.command()
def ping():
    typer.echo("Pong")

@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")

@app.command()
def train(file: Annotated[typer.FileText, "CSV file used for training"]):
    from .cmd.train import TrainSet, TrainData

    trainset = TrainSet()
    
    reader = csv.DictReader(file)

    from rich.progress import track

    for row in track(reader, description="Loading CSV file"):
        trainset.add(TrainData(**row))

    typer.echo(f"Loaded {len(trainset.dataset)} rows")