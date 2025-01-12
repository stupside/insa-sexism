import csv
import typer

from torch import save, Tensor
from numpy import append

from rich.progress import track

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

    vectorizer = TextVectorizer(top_k=10_000)

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

    opts = ModelOptions(
        input_dim=vocab_size,
        output_dim=2,
        hidden_dim=64,
        num_layers=1,
        dropout_rate=0.5,
    )
    model = Model.get(options=opts)

    opts = TrainOptions(
        num_epochs=15,
        batch_size=32,
        learn_rate=1e-3,
        num_workers=0,
        weight_decay=1e-4,
        cross_entropy_weight=[1.0, 2.0],
    )

    fitting = fit(model, options=opts, dataset=trainset)

    metrics: list[MetricCompute] = []
    for _, metric in track(fitting, description="Fitting model", total=opts.num_epochs):
        metrics.append(metric)

    for metric in metrics:
        typer.echo(
            f"F1: {metric.f1}, AUROC: {metric.auroc}, Recall: {metric.recall}, Accuracy: {metric.accuracy}, Precision: {metric.precision}"
        )
        typer.echo(f"Confusion Matrix: {metric.confusion_matrix}")

    typer.echo(f"Saving model to {output}")
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

    typer.echo(f"Sexist: {num_yes}, Nonsexist: {num_no} ({num_yes / num_no:.2f})")
