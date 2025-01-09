import random


from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from numpy import float32


from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom imports
from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat
from cmd.MLP_V1.mlp_tweets_labels_wrapper import MLP_TWEETS_LABELS_WRAPPER


class MLP_PREPOCESS_DATA:
    """
    Stages of preprocessing data for the MLP include :
        1. Formatting and cleaning the data properly based on the initial data structure loaded
        2. Tokenizing and vectorizing the data in order to pass it to the model
    """

    # Brute data as extracted from the dataset
    brute_training_data: list[TrainingDataFormat] = []
    brute_validation_data: list[ValidationDataFormat] = []

    # Formatted data
    formatted_training_data: MLP_TWEETS_LABELS_WRAPPER = MLP_TWEETS_LABELS_WRAPPER()
    formatted_validation_tweets: list[str] = []

    NUMBER_OF_VOTES = 6

    # Vectorization parameters

    def add(self, data: TrainingDataFormat):

        self.training_dataset.append(data)

    def clean(self):

        download("punkt_tab")
        download("stopwords")

        stop_words = set(stopwords.words("english"))

        for data in self.training_dataset:

            sentence = word_tokenize(data.tweet.lower())

            filtered = [word for word in sentence if word.isalnum()]

            filtered = [word for word in filtered if word not in stop_words]

            data.tweet = " ".join(filtered)

    def is_sexist(self, data: TrainingDataFormat):

        yes = 0

        for label in data.labels_task1:
            if label == "YES":
                yes += 1

        if yes >= (self.NUMBER_OF_VOTES / 2):

            return 1

        return 0

    def get_training_set(self) -> TrainingSet:

        training_set = TrainingSet()

        for data in self.training_dataset:
            is_sexist = self.is_sexist(data)
            training_set.add(data.tweet, is_sexist)

        return training_set

    def add(self, tweet: str, label: str):
        self.trainning_labels.append(label)
        self.training_tweets.append(tweet)

    def shuffle(self, seed=123):
        random.seed(seed)
        random.shuffle(self.trainning_labels)
        random.seed(seed)
        random.shuffle(self.training_tweets)

    def __step_1_format_and_clean_training_data(self):
        tweets_lis
        return "hello"

    def __step_2_vectorize(
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
            "decode_error": "strict",
            "strip_accents": "unicode",
            "lowercase": True,
        }

        vectorizer = TfidfVectorizer(**kwargs)

        # Learn vocabulary from training texts and vectorize training texts.
        x_train = vectorizer.fit_transform(self.training_tweets)

        # Vectorize validation texts.
        x_val = vectorizer.transform(self.validation_tweets)

        # Select top 'k' of the vectorized features.
        selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
        selector.fit(x_train, self.trainning_labels)
        x_train = selector.transform(x_train).astype("float32")
        x_val = selector.transform(x_val).astype("float32")
        return x_train, x_val
