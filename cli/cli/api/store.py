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

    def is_sexist(self, data: TrainData) -> bool:
        # Initialize weights for different types of sexism
        sexism_weights = {
            "-": 0.5,
            "DIRECT": 1.0,
            "JUDGEMENTAL": 0.8,
            "REPORTED": 0.7,
        }

        # Initialize demographic weights (optional tuning based on analysis)
        demographic_weights = {
            "gender": {"F": 1.2, "M": 0.8},  # Slight bias towards female annotators
            "age": {"18-22": 1.0, "23-45": 1.0, "46+": 1.0},  # Equal weighting for now
            "ethnicity": {},  # Placeholder for future analysis
            "education": {},  # Placeholder for future analysis
        }

        total_score = 0
        valid_votes = 0

        for idx in range(len(data.annotators)):

            task1_label = data.labels_task1[idx]
            task2_label = data.labels_task2[idx]

            age = data.age_annotators[idx]
            gender = data.gender_annotators[idx]

            # Skip invalid or incomplete annotations
            if task1_label not in ["YES", "NO"] or task2_label not in sexism_weights:
                continue

            vote_score = 0
            valid_votes += 1

            # Calculate sexism score
            if task1_label == "YES":
                vote_score = sexism_weights[task2_label]
            elif task1_label == "NO":
                vote_score = -1  # Slight penalty for NO votes

            # Apply demographic adjustments (if available)
            if gender in demographic_weights["gender"]:
                vote_score *= demographic_weights["gender"][gender]
            if age in demographic_weights["age"]:
                vote_score *= demographic_weights["age"][age]

            total_score += vote_score

        # Normalize score (-1 to 1 range)
        if valid_votes == 0:
            raise ValueError("No valid votes found for this data point.")

        normalized_score = total_score / valid_votes
        print(f"Normalized score: {normalized_score}")

        # Set a decision threshold

        threshold = 0  # Example threshold, tune based on validation

        return normalized_score >= threshold

    def get_training_set(self) -> Trainer:

        training_set = Trainer()

        for data in self.train_set:
            is_sexist = self.is_sexist(data)
            training_set.add_train_data(data.tweet, is_sexist)

        for data in self.test_set:
            training_set.add_test_data(data.tweet)

        return training_set
