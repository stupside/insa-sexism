import csv
import typer


from hydra import compose, initialize
from typing import List, Optional
from omegaconf import DictConfig


from numpy import append
from torch import save

from rich.console import Console
from rich.progress import track

from typing_extensions import Annotated

from .src.loader import read

from .src.vectorizer import TextVectorizer, TextVectorizerOptions

from .src.sets.train import TrainDataSet

from .src.types.train import TrainData

from .src.clean import clean_text
from .src.model import Model, ModelOptions
from .src.trainer import fit, predict, split, validate, TrainOptions

from cli.src.tuning import tune_hyperparams

from .src.types.train import AnnotatorGender


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


def _extract_annotator(d: TrainData, sex: AnnotatorGender, age: str) -> TrainData:

    data = TrainData()

    data.tweet = d.tweet

    # Get indexs of female annotators
    gender_indexes = [
        i for i, gender in enumerate(d.gender_annotators) if gender == sex
    ]
    age_indexes = [i for i, x in enumerate(d.age_annotators) if x == age]

    # Get the intersection of the two lists
    indexs = list(set(gender_indexes).intersection(age_indexes))

    _f_task1_labels = [d.labels_task1[i] for i in indexs]

    data.labels_task1 = _f_task1_labels

    data.age_annotators = [d.age_annotators[i] for i in indexs]
    data.gender_annotators = [d.gender_annotators[i] for i in indexs]

    return data


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

    # Load the model
    console.log("Handling F 18-22")
    f_young_model = Model.get(
        vocab_size=len(vectorizer.vocabulary), options=config.model
    )
    f_young_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        f_young_data = _extract_annotator(data, "F", "18-22")
        f_young_trainset.datas = append(f_young_trainset.datas, f_young_data)
    # Train and validate the model
    trainsubset, valsubset = split(options=config.train, dataset=f_young_trainset)
    fitting = fit(f_young_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(f_young_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Load the model
    console.log("Handling F 23-45")
    f_mid_model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)
    f_mid_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        f_mid_data = _extract_annotator(data, "F", "23-45")
        f_mid_trainset.datas = append(f_mid_trainset.datas, f_mid_data)
    # Train and validate the model
    trainsubset, valsubset = split(options=config.train, dataset=f_mid_trainset)
    fitting = fit(f_mid_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(f_mid_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Load the model
    console.log("Handling F 46+")
    f_old_model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)
    f_old_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        f_old_data = _extract_annotator(data, "F", "46+")
        f_old_trainset.datas = append(f_old_trainset.datas, f_old_data)
    # Train and validate the model
    trainsubset, valsubset = split(options=config.train, dataset=f_old_trainset)
    fitting = fit(f_old_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(f_old_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Load the model
    console.log("Handling M 18-22")
    m_young_model = Model.get(
        vocab_size=len(vectorizer.vocabulary), options=config.model
    )
    m_young_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        m_young_data = _extract_annotator(data, "M", "18-22")
        m_young_trainset.datas = append(m_young_trainset.datas, m_young_data)
    # Train and validate the model
    trainsubset, valsubset = split(options=config.train, dataset=m_young_model)
    fitting = fit(m_young_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(m_young_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Load the model
    console.log("Handling M 23-45")
    m_mid_model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)
    m_mid_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        m_mid_data = _extract_annotator(data, "M", "23-45")
        m_mid_trainset.datas = append(m_mid_trainset.datas, m_mid_data)
    # Train and validate the model
    trainsubset, valsubset = split(options=config.train, dataset=m_mid_trainset)
    fitting = fit(m_mid_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(m_mid_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Load the model
    console.log("Handling M 46+")
    m_old_model = Model.get(vocab_size=len(vectorizer.vocabulary), options=config.model)
    m_old_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        m_old_data = _extract_annotator(data, "M", "46+")
        m_old_trainset.datas = append(m_old_trainset.datas, m_old_data)
    trainsubset, valsubset = split(options=config.train, dataset=trainset)
    # Train and validate the model
    fitting = fit(m_old_model, subset=trainsubset, options=config.train)
    for epoch, metric, loss in fitting:
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")
    validating = validate(m_old_model, subset=valsubset, options=config.train)
    for metric in validating:
        console.log(f"Validation: {metric}")

    # Save the model
    with open(output_file + "_f_young", "wb") as file:
        save(f_young_model.state_dict(), file)

    with open(output_file + "_f_mid", "wb") as file:
        save(f_mid_model.state_dict(), file)

    with open(output_file + "_f_old", "wb") as file:
        save(f_old_model.state_dict(), file)

    with open(output_file + "_m_young", "wb") as file:
        save(m_young_model.state_dict(), file)

    with open(output_file + "_m_mid", "wb") as file:
        save(m_mid_model.state_dict(), file)

    with open(output_file + "_m_old", "wb") as file:
        save(m_old_model.state_dict(), file)

    f_young_model.eval()

    class Prediction:
        ID: str
        pred: int

    predictions: list[Prediction] = []
    for data in track(trainset.datas, description="Predicting with model"):

        f_young_prediction = predict(
            f_young_model, vectorizer=vectorizer, text=data.tweet
        )
        f_mid_prediction = predict(f_mid_model, vectorizer=vectorizer, text=data.tweet)
        f_old_prediction = predict(f_old_model, vectorizer=vectorizer, text=data.tweet)

        m_young_prediction = predict(
            m_young_model, vectorizer=vectorizer, text=data.tweet
        )
        m_mid_prediction = predict(m_mid_model, vectorizer=vectorizer, text=data.tweet)
        m_old_prediction = predict(m_old_model, vectorizer=vectorizer, text=data.tweet)

        votes = [
            f_young_prediction,
            f_mid_prediction,
            f_old_prediction,
            m_young_prediction,
            m_mid_prediction,
            m_old_prediction,
        ]

        prediction = Prediction()
        prediction.ID = data.ID
        prediction.pred = sum(votes) / len(votes)

        predictions.append(prediction)

    with open(output_file + "_predictions", "w") as file:
        writer = csv.writer(file)

        writer.writerow(["ID", "pred"])

        for prediction in predictions:
            writer.writerow([prediction.ID, prediction.pred])


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

    result = tune_hyperparams(trials=100, vectorizer=vectorizer, trainset=trainset)

    console.log(f"Best hyperparameters: {result}")
