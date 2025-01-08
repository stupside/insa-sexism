import random


import nltk


import sklearn


from nltk.corpus import stopwords


from nltk.tokenize import word_tokenize


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_selection import SelectKBest


from sklearn.feature_selection import f_classif


class TrainingSet:

    labels: list[str]
    tweets: list[str]

    def add(self, tweet: str, label: str):
        self.labels.append(label)
        self.tweets.append(tweet)

    def shuffle(self, seed=123):
        random.seed(seed)
        random.shuffle(self.labels)
        random.seed(seed)
        random.shuffle(self.tweets)


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

    dataset: list[TrainData]

    NUMBER_OF_VOTES = 6

    # Vectorization parameters

    # Range (inclusive) of n-gram sizes for tokenizing text.

    NGRAM_RANGE = (1, 2)

    # Limit on the number of features. We use the top 20K features.

    TOP_K = 20000

    # Whether text should be split into word or character n-grams.

    # One of 'word', 'char'.

    TOKEN_MODE = "word"

    # Minimum document/corpus frequency below which a token will be discarded.

    MIN_DOCUMENT_FREQUENCY = 2

    def add(self, data: TrainData):

        self.dataset.append(data)

    def clean(self):

        nltk.download("punkt_tab")

        nltk.download("stopwords")

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

    def ngram_vectorize(self, train_texts, train_labels):
        """Vectorizes texts as n-gram vectors.

        1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

        # Arguments

            train_texts: list, training text strings.

            train_labels: np.ndarray, training labels.

        # Returns

            x_train: vectorized training texts

        """

        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        kwargs = {
            "ngram_range": self.NGRAM_RANGE,  # Use 1-grams + 2-grams.
            "dtype": "int32",
            "strip_accents": "unicode",
            "decode_error": "replace",
            "analyzer": self.TOKEN_MODE,  # Split text into word tokens.
            "min_df": self.MIN_DOCUMENT_FREQUENCY,
        }

        # Instantiate the vectorizer
        vectorizer = TfidfVectorizer(**kwargs)

        # Learn vocabulary from training texts and vectorize training texts.
        x_train = vectorizer.fit_transform(train_texts)

        # Select top 'k' of the vectorized features.
        selector = SelectKBest(f_classif, k=min(self.TOP_K, x_train.shape[1]))

        selector.fit(x_train, train_labels)

        x_train = selector.transform(x_train).astype("float32")

        # return the vectorized training and validation texts
        return x_train


def run(trainset: TrainSet):

    print(f"Loaded {len(trainset.dataset)} rows")

    # Cleaning the data : remove punctuation, stopwords, etc.

    trainset.clean()

    # Get both the training data and the labels associated with it

    training_set = trainset.get_training_set()

    training_set.shuffle(123)

    x_train = trainset.ngram_vectorize(training_set.tweets, training_set.labels)

    print(x_train)

    for data in trainset.dataset:

        print(data.tweet)

    for data in trainset.dataset:

        print(data.labels_task1)
