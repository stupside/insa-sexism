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
from .src.model import EmbeddingModel, ClassifierModel, EmbeddingOptions, ModelOptions
from .src.trainer import fit, predict, split, validate, TrainOptions, fit_embedding


from .src.types.train import AnnotatorGender


app = typer.Typer(
    help="This is a CLI tool detect sexism in text.", no_args_is_help=True
)

console = Console()


class Config:
    model: ModelOptions
    train: TrainOptions
    embedding: EmbeddingOptions
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


def _extract_annotator(
    d: TrainData, sex: AnnotatorGender, age: str
) -> TrainData | None:
    data = TrainData()

    data.tweet = d.tweet

    # Find indexes where both gender and age match
    matching_indexes = [
        i
        for i in range(len(d.gender_annotators))
        if d.gender_annotators[i] == sex and d.age_annotators[i] == age
    ]

    # Extract matching annotations
    if matching_indexes:
        data.annotators = [d.annotators[i] for i in matching_indexes]
        data.gender_annotators = [d.gender_annotators[i] for i in matching_indexes]
        data.age_annotators = [d.age_annotators[i] for i in matching_indexes]
        data.labels_task1 = [d.labels_task1[i] for i in matching_indexes]
        data.labels_task2 = [d.labels_task2[i] for i in matching_indexes]
    else:
        return None

    return data


class Prediction:
    ID: str
    pred: float


def _create_and_train_model(
    name: str,
    gender: str,
    age: str,
    embedding_model: EmbeddingModel,
    trainset: TrainDataSet,
    vectorizer: TextVectorizer,
    config: Config,
    console: Console,
) -> ClassifierModel | None:
    console.log(f"Training model {name} ({gender} {age})")

    # Create filtered dataset
    filtered_trainset = TrainDataSet(vectorizer=vectorizer)
    for data in track(trainset.datas, description="Filtering the train set"):
        filtered_data = _extract_annotator(data, gender, age)
        if filtered_data:
            filtered_trainset.datas = append(filtered_trainset.datas, filtered_data)

    # Check if we have enough data and both classes are represented
    if len(filtered_trainset.datas) < 10:  # Minimum dataset size
        console.log(
            f"Insufficient data for model {name} (only {len(filtered_trainset.datas)} samples)"
        )
        return None

    # Count positive and negative samples
    pos_samples = sum(
        1
        for data in filtered_trainset.datas
        for label in data.labels_task1
        if label == "YES"
    )
    neg_samples = sum(
        1
        for data in filtered_trainset.datas
        for label in data.labels_task1
        if label == "NO"
    )

    if pos_samples == 0 or neg_samples == 0:
        console.log(
            f"Skipping model {name}: Missing {'positive' if pos_samples == 0 else 'negative'} samples"
        )
        return None

    # Create model with shared embedding
    model = ClassifierModel.get(
        vocab_size=len(vectorizer.vocabulary),
        options=config.model,
        embedding_model=embedding_model,
    )

    # Train and validate
    trainsubset, valsubset = split(options=config.train, dataset=filtered_trainset)

    for epoch, metric, loss in fit(model, subset=trainsubset, options=config.train):
        console.log(f"Epoch {epoch}: {metric} - Loss: {loss}")

    for metric in validate(model, subset=valsubset, options=config.train):
        console.log(f"Validation: {metric}")

    return model


@app.command()
def train_embedding(
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    overrides: Optional[List[str]] = typer.Argument(None),
):
    """Train the embedding model separately"""

    # Load configuration with embedding-specific defaults
    config: Config = _compose("./", "model.yaml", overrides)

    # Load configuration and setup
    vectorizer = TextVectorizer(options=config.vectorizer, load=True)

    # Load and prepare dataset
    trains = read(train_file)
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    for data in track(trainset.datas, description="Cleaning the train set"):
        data.tweet = clean_text(data.tweet)

    # Create new embedding model using model.embedding_dim
    embedding_model = EmbeddingModel(
        vocab_size=len(vectorizer.vocabulary),
        embedding_dim=config.embedding.embedding_dim,
    )

    trainsubset, _ = split(options=config.train, dataset=trainset)

    for epoch, loss in fit_embedding(embedding_model, config.embedding, trainsubset):
        console.log(f"Embedding training epoch {epoch}: Loss = {loss:.4f}")

    # Save the trained embedding
    save(embedding_model.state_dict(), config.embedding.checkpoint_path)
    console.log(f"Saved embedding to {config.embedding.checkpoint_path}")


@app.command()
def train_model(
    train_file: Annotated[typer.FileText, typer.Option(encoding="UTF-8")],
    output_file: Annotated[str, typer.Option(help="File to save the model to")],
    overrides: Optional[List[str]] = typer.Argument(None),
):
    # Load configuration and setup
    config: Config = _compose("./", "model.yaml", overrides)
    vectorizer = TextVectorizer(options=config.vectorizer, load=True)

    # Load and prepare dataset
    trains = read(train_file)
    trainset = TrainDataSet(vectorizer=vectorizer)
    for train in trains:
        trainset.datas = append(trainset.datas, TrainData(**train))

    for data in track(trainset.datas, description="Cleaning the train set"):
        data.tweet = clean_text(data.tweet)

    # Load and freeze pre-trained embedding model
    embedding_model = EmbeddingModel.load(
        options=config.embedding, vocab_size=len(vectorizer.vocabulary)
    )

    embedding_model.freeze()  # Freeze embedding parameters

    console.log("Loaded and frozen pre-trained embedding")

    # Define model configurations
    model_configs = [
        ("f_young", "F", "18-22"),
        ("f_mid", "F", "23-45"),
        ("f_old", "F", "46+"),
        ("m_young", "M", "18-22"),
        ("m_mid", "M", "23-45"),
        ("m_old", "M", "46+"),
    ]

    # Train all models
    models: dict[str, ClassifierModel] = {}
    for name, gender, age in model_configs:
        model = _create_and_train_model(
            name=name,
            gender=gender,
            age=age,
            embedding_model=embedding_model,
            trainset=trainset,
            vectorizer=vectorizer,
            config=config,
            console=console,
        )

        if model is not None:  # Only save and use valid models
            models[name] = model
            # Save model
            with open(f"{output_file}_{name}", "wb") as file:
                save(model.state_dict(), file)
        else:
            console.log(f"Skipping model {name} due to insufficient data")

    # Update prediction logic to handle missing models
    if not models:
        console.log("No models could be trained due to insufficient data")
        return

    # Make predictions
    predictions: list[Prediction] = []
    for data in track(trainset.datas, description="Predicting with models"):
        votes = [
            predict(model, vectorizer=vectorizer, text=data.tweet)
            for model in models.values()
        ]

        prediction = Prediction()

        prediction.ID = data.ID
        prediction.pred = sum(votes) / len(votes)

        predictions.append(prediction)

    # Save predictions
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
