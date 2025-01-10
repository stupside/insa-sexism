from typing import Literal

from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer

import re

import emoji

from .trainer import Trainer

type LabelTask1 = Literal["YES", "NO"]
type LabelTask2 = Literal["DIRECT", "JUDGEMENTAL", "REPORTED", "-"]

type AnnotatorGender = Literal["F", "M"]


class TestData:
    ID: int

    tweet: str

    # Annotation
    annotators: list[str]
    age_annotators: list[str]
    gender_annotators: list[AnnotatorGender]
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
    gender_annotators: list[AnnotatorGender]
    countries_annotators: list[str]
    ethnicities_annotators: list[str]
    study_levels_annotators: list[str]

    # Labels
    labels_task1: list[LabelTask1]  # YES or NO
    labels_task2: list[LabelTask2]  # DIRECT, JUDGEMENTAL, REPORTED, -

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
        download("wordnet")
        download("punkt_tab")
        download("stopwords")

        stop_words = set(stopwords.words("english"))

        stemmer = LancasterStemmer()
        lemmatizer = WordNetLemmatizer()

        for data in self.train_set:
            # Convert emojis to text
            tweet = emoji.demojize(data.tweet)

            # Remove URLs
            tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

            # Remove mentions but keep the text
            tweet = re.sub(r"@", "", tweet)

            # Remove hashtags but keep the text
            tweet = re.sub(r"#", "", tweet)

            # Convert to lowercase and tokenize
            tokens = word_tokenize(tweet.lower())

            filtered = [stemmer.stem(word) for word in tokens]
            filtered = [lemmatizer.lemmatize(word) for word in tokens]

            # Remove stopwords
            filtered = [word for word in filtered if word not in stop_words]

            data.tweet = " ".join(filtered)

        # for data in self.test_set:
        #     # Convert emojis to text
        #     tweet = emoji.demojize(data.tweet)

        #     # Remove URLs
        #     tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

        #     # Remove mentions but keep the text
        #     tweet = re.sub(r"@", "", tweet)

        #     # Remove hashtags but keep the text
        #     tweet = re.sub(r"#", "", tweet)

        #     # Convert to lowercase and tokenize
        #     tokens = word_tokenize(tweet.lower())

        #     filtered = [stemmer.stem(word) for word in tokens if word.isalnum()]
        #     filtered = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]

        #     # Remove stopwords
        #     filtered = [word for word in filtered if word not in stop_words]

        #     data.tweet = " ".join(filtered)

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
