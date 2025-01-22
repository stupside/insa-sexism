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

from .src.clean import clean_text
from .src.model import Model, ModelOptions
from .src.trainer import fit, predict, split, validate, TrainOptions

from cli.src.tuning import tune_hyperparams

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

    # Create the train set
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        traindata = TrainData(**train)
        trainset.datas = append(trainset.datas, traindata)

    # Clean the train set
    for data in track(trainset.datas, description="Cleaning the train set"):
        data.tweet = clean_text(data.tweet)

    trainsubset, valsubset = split(options=config.train, dataset=trainset)

    # Train the model
    fitting = fit(model, subset=trainsubset, options=config.train)
    for epoch, metric in track(
        fitting, description="Training the model", total=config.train.num_epochs
    ):
        console.log(f"Epoch {epoch}: {metric}")

    validating = validate(model, subset=valsubset, options=config.train)
    for metric in track(
        validating, description="Validating the model", total=len(valsubset)
    ):
        console.log(f"Validation metrics: {metric}")

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

    # Create the test set
    testset = TestDataSet(vectorizer=vectorizer)
    for test in tests:
        testset.datas = append(testset.datas, TestData(**test))

    # Create a cleaned test set only for predictions
    cleaned_testset = TestDataSet(vectorizer=vectorizer)
    cleaned_testset.datas = testset.datas.copy()
    for test in track(cleaned_testset.datas, description="Cleaning the test set"):
        test.tweet = clean_text(test.tweet)

    # Predict on the test set
    predictions: list[Tensor] = []
    for tweet in track(
        cleaned_testset.datas,
        description="Predicting with model",
        total=len(cleaned_testset.datas),
    ):
        predictions.append(predict(model, vectorizer=vectorizer, text=tweet.tweet))

    # Save the predictions
    with open(output_path, "w") as file:
        writer = csv.writer(file)

        writer.writerow(["tweet", "prediction"])

        for idx, prediction in track(
            predictions,
            description="Saving predictions to file",
            total=len(predictions),
        ):
            data = testset.datas[idx]
            writer.writerow(
                [
                    data.tweet,
                    prediction.item(),
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

    # Create the train set
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        traindata = TrainData(**train)
        trainset.datas = append(trainset.datas, traindata)

    # Clean the train set
    for data in track(trainset.datas, description="Cleaning the train set"):
        data.tweet = clean_text(data.tweet)

    # Augment the train set
    # augmented_trainset = TrainDataSet(vectorizer=vectorizer)
    # for data in track(
    #     trainset.datas,
    #     description="Augmenting the train set",
    #     total=len(trainset.datas),
    # ):
    #     augmented_data = augment_with_synonyms(
    #         tokens=data.tweet.split(), num_replacements=2, max_augmented=5
    #     )
    #     for augmented in augmented_data:
    #         augmented_trainset.datas = append(
    #             augmented_trainset.datas, TrainData(tweet=" ".join(augmented))
    #         )
    #         print(f"Augmented: {' '.join(augmented)}")

    # Fit the vectorizer
    vectorizer.fit([tweet.tweet for tweet in trainset.datas], save=True)
    console.log(
        f"Dictionnary saved to {config.vectorizer.dictionnary_path} with size {len(vectorizer.vocabulary)}"
    )


@app.command()
def tune(
    # Files
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    # Hydra options
    overrides: Optional[List[str]] = typer.Argument(None),
):
    config: Config = _compose("./", "model.yaml", overrides)

    # Create a vectorizer
    vectorizer = TextVectorizer(options=config.vectorizer, load=True)

    # Load the train set
    trains = read(train_file)

    # Create the train set
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        traindata = TrainData(**train)
        trainset.datas = append(trainset.datas, traindata)

    # Clean the train set
    for data in track(trainset.datas, description="Cleaning the train set"):
        data.tweet = clean_text(data.tweet)

    result = tune_hyperparams(trials=15, vectorizer=vectorizer, trainset=trainset)

    console.log(f"Best hyperparameters: {result}")
