from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .trainer import Trainer


class TestData:
    ID: int

    tweet: str

    # Annotation
    annotators: list[str]
    age_annotators: list[str]
    gender_annotators: list[str]
    countries_annotators: list[str]
    ethnicities_annotators: list[str]
    study_levels_annotators: list[str]

    def __init__(self, **fields):
        self.__dict__.update(**fields)


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


class DataStore:

    test_set: list[TestData] = []
    train_set: list[TrainData] = []

    NUMBER_OF_VOTES = 6

    # Vectorization parameters

    def add_train_data(self, data: TrainData):
        self.train_set.append(data)

    def add_test_data(self, data: TestData):
        self.test_set.append(data)

    def clean_train_data(self):
        download("punkt_tab")
        download("stopwords")

        stop_words = set(stopwords.words("english"))

        for data in self.train_set:

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

    def get_training_set(self) -> Trainer:

        training_set = Trainer()

        for data in self.train_set:
            is_sexist = self.is_sexist(data)
            training_set.add_train_data(data.tweet, is_sexist)

        for data in self.test_set:
            training_set.add_test_data(data.tweet)

        return training_set
