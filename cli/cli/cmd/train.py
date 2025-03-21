import random

from numpy import float32

from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


class TrainingSet:

    labels: list[str] = []
    tweets: list[str] = []

    def add(self, tweet: str, label: str):
        self.labels.append(label)
        self.tweets.append(tweet)

    def shuffle(self, seed=123):
        random.seed(seed)
        random.shuffle(self.labels)
        random.seed(seed)
        random.shuffle(self.tweets)

    def vectorize(
        self,
        top_k: int,
        token_mode: str,
        ngram_range: tuple[int, int],
        min_document_frequency: int,
    ):
        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        kwargs = {
            "dtype": float32,
            "min_df": min_document_frequency,
            "analyzer": token_mode,
            "ngram_range": ngram_range,
            "decode_error": "replace",
            "strip_accents": "unicode",
        }

        # Instantiate the vectorizer
        vectorizer = TfidfVectorizer(**kwargs)

        # Learn vocabulary from training texts and vectorize training texts.
        x_train = vectorizer.fit_transform(self.tweets)

        # Select top 'k' of the vectorized features.
        selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))

        selector.fit(x_train, self.labels)

        x_train = selector.transform(x_train).astype("float32")

        # return the vectorized training
        return x_train


# tweet,annotators,gender_annotators,age_annotators,ethnicities_annotators,study_levels_annotators,countries_annotators,labels_task1,labels_task2,ID
class TrainData:

    ID: int

    tweet: str

    # Annotation

    annotators: list[str]

    age_annotators: list[str]

    gender_annotators: list[str]

    countries_annotators: list[str]

    ethnicities_annotators: list[str]

    study_levels_annotators: list[str]

    # Labels

    labels_task1: list[str]

    labels_task2: list[str]

    def __init__(self, **fields):

        self.__dict__.update(**fields)


class TrainSet:

    dataset: list[TrainData] = []

    NUMBER_OF_VOTES = 6

    # Vectorization parameters

    def add(self, data: TrainData):

        self.dataset.append(data)

    def clean(self):

        download("punkt_tab")
        download("stopwords")

        stop_words = set(stopwords.words("english"))

        for data in self.dataset:

            sentence = word_tokenize(data.tweet.lower())

            filtered = [word for word in sentence if word.isalnum()]

            filtered = [word for word in filtered if word not in stop_words]

            data.tweet = " ".join(filtered)

    def is_sexist(self, data: TrainData):

        yes = 0

        for label in data.labels_task1:
            if label == "YES":
                yes += 1

        if yes >= (self.NUMBER_OF_VOTES / 2):

            return 1

        return 0

    def get_training_set(self) -> TrainingSet:

        training_set = TrainingSet()

        for data in self.dataset:
            is_sexist = self.is_sexist(data)
            training_set.add(data.tweet, is_sexist)

        return training_set

    def _get_last_layer_units_and_activation(num_classes):
        """Gets the # units and activation function for the last network layer.

        # Arguments
            num_classes: int, number of classes.

        # Returns
            units, activation values.
        """
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        return units, activation

    def mlp_model(self, layers, units, dropout_rate, input_shape, num_classes):
        """Creates an instance of a multi-layer perceptron model.

        # Arguments
            layers: int, number of `Dense` layers in the model.
            units: int, output dimension of the layers.
            dropout_rate: float, percentage of input to drop at Dropout layers.
            input_shape: tuple, shape of input to the model.
            num_classes: int, number of output classes.

        # Returns
            An MLP model instance.
        """
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)
        model = models.Sequential()
        model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

        for _ in range(layers - 1):
            model.add(Dense(units=units, activation="relu"))
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(units=op_units, activation=op_activation))
        return model


def run(
    trainset: TrainSet,
    seed: int,
    top_k: int,
    token_mode: str,
    ngram_range: tuple[int, int],
    min_document_frequency: int,
):

    # Cleaning the data : remove punctuation, stopwords, etc.
    trainset.clean()

    # Get both the training data and the labels associated with it
    training_set = trainset.get_training_set()

    training_set.shuffle(seed)

    vector = training_set.vectorize(
        top_k, token_mode, ngram_range, min_document_frequency
    )

    print(vector)
