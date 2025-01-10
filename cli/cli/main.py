import csv
import typer

from typing_extensions import Annotated

from matplotlib import pyplot
from tensorflow import summary

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
    fit_log_verbosity: int = 1,
    # Model
    output: str = "./out/model.keras",
    # Training
    layers: int = 2,
    fit_epochs: int = 30,
    ngram_top_k: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    dropout_rate: float = 0.2,
    fit_batch_size: int = 64,
    kfolds_n_splits: int = 10,
    mlp_dense_units: int = 32,
    adam_learning_rate: float = 5e-4,
    early_stopping_patience: int = 3,
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

    writer = summary.create_file_writer(output + ".logs")

    writer.set_as_default()

    model, predictions, accuracy = training_set.train_ngram_model(
        layers=layers,
        fit_epochs=fit_epochs,
        ngram_top_k=ngram_top_k,
        ngram_range=ngram_range,
        dropout_rate=dropout_rate,
        fit_batch_size=fit_batch_size,
        kfolds_n_splits=kfolds_n_splits,
        mlp_dense_units=mlp_dense_units,
        fit_log_verbosity=fit_log_verbosity,
        adam_learning_rate=adam_learning_rate,
        early_stopping_patience=early_stopping_patience,
    )

    print(accuracy)

    model.summary()

    model.save(output)

    with open(output + ".predictions", "w") as f:
        writer = csv.writer(f)

        writer.writerow(["tweet", "prediction"])

        for idx, prediction in enumerate(predictions):
            writer.writerow(
                [
                    trainset.test_set[idx].tweet,
                    prediction,
                ]
            )


@app.command()
def analyze_train_set(
    train: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
):
    from .api.store import DataStore, TrainData

    trainset = DataStore()

    from ast import literal_eval

    # Load the training data
    for row in csv.DictReader(train):
        # Convert string representations of arrays to actual arrays
        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # Keep original string if parsing fails

        trainset.add_train_data(TrainData(**row))

    trainset.clean_train_data()

    trainer = trainset.get_training_set()

    # Count label equals to 1
    sexist = 0
    non_sexist = 0
    for label in trainer.train_labels:
        if label == 1:
            sexist += 1
        else:
            non_sexist += 1

    pyplot.bar(
        ["sexist", "non-sexist"], [sexist, non_sexist], label="Sexist vs Non-Sexist"
    )

    pyplot.show()
