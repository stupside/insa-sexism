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
    fit_log_verbosity: int = 1,
    # Model
    output: str = "./out/model.keras",
    # Training
    layers: int = 2,
    fit_epochs: int = 100,
    ngram_top_k: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
    ngram_min_df: int = 3,
    dropout_rate: float = 0.3,
    fit_batch_size: int = 256,
    kfolds_n_splits: int = 10,
    mlp_dense_units: int = 64,
    ngram_token_mode: str = "word",
    adam_learning_rate: float = 1e-4,
    early_stopping_patience: int = 10,
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

    model, predictions, accuracy = training_set.train_ngram_model(
        layers=layers,
        fit_epochs=fit_epochs,
        ngram_top_k=ngram_top_k,
        ngram_range=ngram_range,
        ngram_min_df=ngram_min_df,
        dropout_rate=dropout_rate,
        fit_batch_size=fit_batch_size,
        kfolds_n_splits=kfolds_n_splits,
        mlp_dense_units=mlp_dense_units,
        ngram_token_mode=ngram_token_mode,
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
