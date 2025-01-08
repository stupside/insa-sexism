from numpy import float64, float32, array, append, random, ndarray

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow.python.keras import models
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import Dropout
# from tensorflow.python.keras.optimizer_v2.adam import Adam

from keras._tf_keras.keras import models, Input
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.optimizers import Adam


class Trainer:

    test_tweets: ndarray = array([], dtype=str)
    train_tweets: ndarray = array([], dtype=str)

    train_labels: ndarray = array([], dtype=int)

    def add_test_data(self, tweet: str):
        self.test_tweets = append(self.test_tweets, tweet)

    def add_train_data(self, tweet: str, label: int):
        self.train_tweets = append(self.train_tweets, tweet)
        self.train_labels = append(self.train_labels, label)

    def shuffle_train_data(self, seed: int):
        random.seed(seed)
        perm = random.permutation(len(self.train_labels))
        self.train_labels = self.train_labels[perm]
        self.train_tweets = self.train_tweets[perm]

    def _get_num_classes(self) -> int:
        num_classes = max(self.train_labels) + 1
        missing_classes = [i for i in range(num_classes) if i not in self.train_labels]
        if len(missing_classes):
            raise ValueError(
                "Missing samples with label value(s) "
                "{missing_classes}. Please make sure you have "
                "at least one sample for every label value "
                "in the range(0, {max_class})".format(
                    missing_classes=missing_classes, max_class=num_classes - 1
                )
            )

        if num_classes <= 1:
            raise ValueError(
                "Invalid number of labels: {num_classes}."
                "Please make sure there are at least two classes "
                "of samples".format(num_classes=num_classes)
            )

        return num_classes

    def _ngram_vectorize(
        self,
        top_k: int,
        token_mode: str,
        ngram_range: tuple[int, int],
        min_document_frequency: int,
    ):
        # Validate input data
        if len(self.train_tweets) == 0:
            raise ValueError("Training texts cannot be empty")
        if len(self.train_labels) == 0:
            raise ValueError("Training labels cannot be empty")
        if len(self.test_tweets) == 0:
            raise ValueError("Validation texts cannot be empty")

        kwargs = {
            "dtype": float64,
            "min_df": min_document_frequency,
            "analyzer": token_mode,
            "ngram_range": ngram_range,
            "decode_error": "replace",
            "strip_accents": "unicode",
        }

        vectorizer = TfidfVectorizer(**kwargs)

        # First fit and transform the training data
        x_train = vectorizer.fit_transform(self.train_tweets)

        # Then transform the validation texts using the fitted vectorizer
        x_val = vectorizer.transform(self.test_tweets)

        # Select top 'k' features
        selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1])).fit(
            x_train, self.train_labels
        )

        x_val = selector.transform(x_val).astype("float32")
        x_train = selector.transform(x_train).astype(dtype=float32)

        return x_train, x_val

    @staticmethod
    def _get_last_layer_units_and_activation(num_classes: int):
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        return units, activation

    def _get_mlp_model(
        self, units: int, input_shape, layers: int, dropout_rate: int, num_classes: int
    ):
        op_units, op_activation = Trainer._get_last_layer_units_and_activation(
            num_classes
        )

        inputs = Input(shape=input_shape)
        dropout = Dropout(rate=dropout_rate)(inputs)

        for _ in range(layers - 1):
            dropout = Dense(units=units, activation="relu")(dropout)
            dropout = Dropout(rate=dropout_rate)(dropout)

        outputs = Dense(units=op_units, activation=op_activation)(dropout)

        model = models.Model(inputs=inputs, outputs=outputs)

        return model

    def train_ngram_model(
        self,
        # Debug
        verbosity: int,
        # Vectorization parameters
        top_k: int,
        token_mode: str,
        ngram_range: tuple[int, int],
        min_document_frequency: int,
        # Layer parameters
        units: int,
        layers: int,
        epochs: int,
        batch_size: int,
        dropout_rate: float,
        learning_rate: float,
    ):
        # Validate data before training
        if len(self.train_tweets) == 0:
            raise ValueError("No training data available")
        if len(self.test_tweets) == 0:
            raise ValueError("No test data available")
        if len(self.train_labels) != len(self.train_tweets):
            raise ValueError(
                "Number of training labels must match number of training tweets"
            )

        # Get number of classes from training data only
        num_classes = self._get_num_classes()

        # Vectorize texts.
        x_train, x_val = self._ngram_vectorize(
            top_k=top_k,
            token_mode=token_mode,
            ngram_range=ngram_range,
            min_document_frequency=min_document_frequency,
        )

        # Create model instance.
        model = self._get_mlp_model(
            units=units,
            layers=layers,
            input_shape=x_train.shape[1:],
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Compile model with learning parameters.
        if num_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"

        # Update Adam optimizer initialization
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

        # Train model without validation data
        history = model.fit(
            x_train,
            self.train_labels,
            # validation_data=(x_val, self.test_labels),
            epochs=epochs,
            verbose=verbosity,
            batch_size=batch_size,
        )

        # Get predictions for test data
        predictions = model.predict(x_val)

        return (
            model,
            predictions,
            {
                "training_loss": history.history["loss"][-1],
                "training_accuracy": history.history["acc"][-1],
            },
        )
