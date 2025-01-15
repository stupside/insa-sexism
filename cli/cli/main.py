import csv
import typer


from hydra import compose, initialize
from typing import List, Optional
from omegaconf import DictConfig


from numpy import append
from torch import save, load, Tensor

from rich.console import Console
from rich.progress import track

from typing_extensions import Annotated

from .src.loader import read

from .src.vectorizer import TextVectorizer, TextVectorizerOptions

from .src.sets.test import TestDataSet
from .src.sets.train import TrainDataSet

from .src.types.test import TestData
from .src.types.train import TrainData

from .src.model import Model, ModelOptions
from .src.trainer import fit, predict, TrainOptions


app = typer.Typer(
    help="This is a CLI tool detect sexism in text.", no_args_is_help=True
)

console = Console()


class Config:
    model: ModelOptions
    train: TrainOptions
    vectorizer: TextVectorizerOptions


def _compose(
    config_path: str, config_name: str, overrides: Optional[List[str]]
) -> DictConfig:
    with initialize(config_path=config_path):
        cfg = compose(
            config_name=config_name, overrides=overrides, return_hydra_config=True
        )

        return cfg

    raise ValueError("Could not load configuration")


@app.command()
def train(
    # Files
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    # Output
    output_file: Annotated[str, typer.Option(help="File to save the model to")],
    # Hydra options
    overrides: Optional[List[str]] = typer.Argument(None),
):
    # Load the configuration
    config: Config = _compose("./", "model.yaml", overrides)

    # Load vectorizer
    vectorizer = TextVectorizer(options=config.vectorizer, load=True)

    # Load the model
    model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)

    # Load the train set
    trains = read(train_file)
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    # Train the model
    train_metrics, val_metrics = fit(model, dataset=trainset, options=config.train)

    # Print the metrics
    console.log(f"Train metrics: {train_metrics}")
    console.log(f"Validation metrics: {val_metrics}")

    # Save the model
    with open(output_file, "wb") as file:
        save(model.state_dict(), file)

    console.log(f"Model saved to {output_file}")


@app.command()
def test(
    # Files
    test_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    model_file: Annotated[str, typer.Option(help="Path to the model file")],
    # Output file path
    output_path: Annotated[str, typer.Option(help="File to save the predictions to")],
    # Hydra options
    overrides: Optional[List[str]] = typer.Argument(None),
):
    # Load the configuration
    config: Config = _compose("./", "model.yaml", overrides)

    # Create a vectorizer
    vectorizer = TextVectorizer(options=config.vectorizer, load=True)
    console.log(f"Vectorizer loaded with {len(vectorizer.vocabulary)} words")

    # Load the model
    model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)
    # Load the model weights
    model.load_state_dict(state_dict=load(model_file, weights_only=True))

    # Set the model to evaluation mode
    model.eval()

    # Load the datasets
    tests = read(test_file)
    testset = TestDataSet(vectorizer=vectorizer)
    for test in tests:
        testset.datas = append(testset.datas, TestData(**test))

    # Predict on the test set
    predictions: list[Tensor] = []
    for tweet in track(
        testset.datas, description="Predicting with model", total=len(testset.datas)
    ):
        predictions.append(predict(model, vectorizer=vectorizer, text=tweet.tweet))

    # Save the predictions
    with open(output_path, "w") as file:
        writer = csv.writer(file)

        writer.writerow(["tweet", "prediction"])

        for idx, prediction in track(
            enumerate(predictions),
            description="Saving predictions to file",
            total=len(predictions),
        ):
            writer.writerow(
                [
                    testset.datas[idx].tweet,
                    "YES" if prediction == 1 else "NO",
                ]
            )


@app.command()
def vectorize(
    # Files
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    # Hydra options
    overrides: Optional[List[str]] = typer.Argument(None),
):
    # Load the configuration
    config: Config = _compose("./", "model.yaml", overrides)

    # Create a vectorizer
    vectorizer = TextVectorizer(options=config.vectorizer, load=False)

    # Load the train set
    trains = read(train_file)
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    # Fit the vectorizer
    vectorizer.fit([tweet.tweet for tweet in trainset.datas], save=True)
    console.log(
        f"Dictionnary saved to {config.vectorizer.dictionnary_path} with size {len(vectorizer.vocabulary)}"
    )


@app.command()
def check(
    # Files
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
):
    trains = read(train_file)
    trainset = TrainDataSet(vectorizer=None)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    num_no = 0
    num_yes = 0
    for _, data in enumerate(trainset.datas):
        sexist = trainset.check_is_sexist(data)
        if not sexist:
            num_no += 1
        else:
            num_yes += 1

    console.log(f"Sexist: {num_yes}, Nonsexist: {num_no} ({num_yes / num_no:.2f})")
