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
    # Sets
    test: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    train: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    # Debug
    verbosity: int = 1,
    # Output
    output: str = "./out/model.keras",
    # Vectorization parameters
    top_k: int = 10000,
    token_mode: str = "word",
    ngram_range: tuple[int, int] = (1, 2),
    min_document_frequency: int = 3,
    # Layer parameters
    units: int = 64,
    layers: int = 2,
    epochs: int = 100,
    batch_size: int = 256,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    early_stopping_patience: int = 10,
    # Others
    folds: int = 10,
):

    from ast import literal_eval

    from .api.store import DataStore, TrainData, TestData

    trainset = DataStore()

    from rich.progress import track

    # Load the training data
    for row in track(csv.DictReader(train), description="Loading training data"):
        # Convert string representations of arrays to actual arrays
        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # Keep original string if parsing fails

        trainset.add_train_data(TrainData(**row))

    # Load the testing data
    for row in track(csv.DictReader(test), description="Loading testing data"):
        # Convert string representations of arrays to actual arrays
        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval
                except (ValueError, SyntaxError):
                    pass

        trainset.add_test_data(TestData(**row))

    # Cleaning the data : remove punctuation, stopwords, etc.
    trainset.clean_train_data()

    # Get both the training data and the labels associated with it
    training_set = trainset.get_training_set()

    model, _, accuracy = training_set.train_ngram_model(
        # Debug
        verbosity=verbosity,
        # Vectorization parameters
        top_k=top_k,
        token_mode=token_mode,
        ngram_range=ngram_range,
        min_document_frequency=min_document_frequency,
        # Layer parameters
        units=units,
        layers=layers,
        epochs=epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        folds=folds,
        early_stopping_patience=early_stopping_patience,
    )

    print(accuracy)
    model.summary()

    model.save(output)
