import csv
import typer

from torch import save, Tensor
from numpy import append

from rich.progress import track
from rich.console import Console

from typing_extensions import Annotated

from .src.loader import load

from .src.vectorizer import TextVectorizer

from .src.sets.test import TestDataSet
from .src.sets.train import TrainDataSet

from .src.types.test import TestData
from .src.types.train import TrainData

from .src.model import Model, ModelOptions
from .src.metric import MetricCompute
from .src.trainer import fit, predict, TrainOptions


app = typer.Typer(
    help="This is a CLI tool detect sexism in text.", no_args_is_help=True
)

console = Console()


@app.command()
def ping():
    typer.echo("Pong")


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def train(
    test: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    train: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    output: Annotated[str, typer.Option(help="File to save the model to")],
):
    # Initialize vectorizer with just max_length
    vectorizer = TextVectorizer(max_length=248)

    tests = load(test)
    testset = TestDataSet(vectorizer=vectorizer)
    for test in tests:
        testset.datas = append(testset.datas, TestData(**test))

    trains = load(train)
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    vectorizer.fit([tweet.tweet for tweet in trainset.datas])

    # Get vocabulary size for input dimension
    vocab_size = len(vectorizer.vocabulary)
    console.log(f"Vocabulary size: {vocab_size}")

    # Save the vocabulary to a file
    with open(output + ".vocab", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "index"])

        for word, index in vectorizer.vocabulary.items():
            writer.writerow([word, index])

    opts = ModelOptions(
        input_dim=vocab_size,
        output_dim=1,
        layer_dims=[256, 128, 64],
        dropout_rate=0.5,
        embedding_dim=768,
        vocab_size=vocab_size,
    )
    model = Model.get(options=opts)

    opts = TrainOptions(
        seed=150,
        num_epochs=16,
        batch_size=16,
        learn_rate=1e-4,
        num_workers=0,
        weight_decay=1e-5,
        train_val_split=0.85,
    )

    fitting = fit(model, options=opts, dataset=trainset)

    val_metrics: list[MetricCompute] = []
    train_metrics: list[MetricCompute] = []
    for _, train_metric, val_metric in track(
        fitting, description="Fitting model", total=opts.num_epochs
    ):
        val_metrics.append(val_metric)
        train_metrics.append(train_metric)

    for idx, (train_metric, val_metric) in enumerate(zip(train_metrics, val_metrics)):
        console.log(f"Epoch {idx + 1}/{opts.num_epochs}")
        console.log(f"Train {train_metric}")
        console.log(f"Validation {val_metric}")

    console.log(f"Saving model to {output}")
    save(model.state_dict(), output)

    predictions: list[Tensor] = []
    for tweet in track(
        testset.datas, description="Predicting", total=len(testset.datas)
    ):
        predictions.append(predict(model, vectorizer, tweet.tweet))

    with open(output + ".test", "w") as file:
        writer = csv.writer(file)

        writer.writerow(["tweet", "prediction"])

        for idx, prediction in track(
            enumerate(predictions),
            description="Saving predictions",
            total=len(predictions),
        ):
            writer.writerow(
                [
                    testset.datas[idx].tweet,
                    "YES" if prediction == 1 else "NO",
                ]
            )


@app.command()
def check_balance(
    train: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
):
    trains = load(train)
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
