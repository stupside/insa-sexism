from numpy import float32, array, append, ndarray

from tensorflow import data

from keras._tf_keras.keras import models, Input
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.layers import TextVectorization

import numpy as np


class Trainer:

    test_tweets: ndarray = array([], dtype=str)
    train_tweets: ndarray = array([], dtype=str)

    train_labels: ndarray = array([], dtype=int)

    def add_test_data(self, tweet: str):
        self.test_tweets = append(self.test_tweets, tweet)

    def add_train_data(self, tweet: str, label: int):
        self.train_tweets = append(self.train_tweets, tweet)
        self.train_labels = append(self.train_labels, label)

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
        ngram_range: tuple[int, int],
        ngram_top_k: int,
    ):
        # Validate input data
        if len(self.train_tweets) == 0:
            raise ValueError("Training texts cannot be empty")
        if len(self.train_labels) == 0:
            raise ValueError("Training labels cannot be empty")
        if len(self.test_tweets) == 0:
            raise ValueError("Validation texts cannot be empty")

        # Create and configure the TextVectorization layer
        vectorizer = TextVectorization(
            max_tokens=ngram_top_k,
            split="whitespace",
            output_mode="count",  # Changed from tf-idf to count
            ngrams=ngram_range,
            output_sequence_length=None,
            standardize="lower_and_strip_punctuation",
        )

        # Adapt the vectorizer to the training data
        vectorizer.adapt(self.train_tweets)

        print(vectorizer.get_vocabulary()[:100])

        # Transform the texts to vectors
        x_val = vectorizer(self.test_tweets)
        x_train = vectorizer(self.train_tweets)

        # Convert to numpy arrays directly
        x_val = x_val.numpy().astype(float32)
        x_train = x_train.numpy().astype(float32)

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
        self,
        layers: int,
        input_shape,
        num_classes: int,
        dropout_rate: int,
        mlp_dense_units: int,
    ):
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)

        inputs = Input(shape=input_shape)

        # Add BatchNormalization at the start
        normalization = inputs
        normalization = BatchNormalization()(normalization)

        for i in range(layers):
            # Decrease units gradually in deeper layers
            layer_units = mlp_dense_units // (2**i)
            if layer_units < 32:  # Minimum number of units
                layer_units = 32

            normalization = Dense(
                units=layer_units,
                activation="relu",
                kernel_regularizer=l2(0.01),  # L2 regularization
            )(normalization)
            normalization = BatchNormalization()(normalization)
            normalization = Dropout(rate=dropout_rate)(normalization)

        outputs = Dense(units=op_units, activation=op_activation)(normalization)

        model = models.Model(inputs=inputs, outputs=outputs)

        return model

    def train_ngram_model(
        self,
        layers: int,
        fit_epochs: int,
        ngram_range: tuple[int, int],
        ngram_top_k: int,
        dropout_rate: float,
        fit_batch_size: int,
        mlp_dense_units: int,
        kfolds_n_splits: int,
        fit_log_verbosity: int,
        adam_learning_rate: float,
        early_stopping_patience: int,
    ) -> tuple[models.Sequential, ndarray, dict[str, float]]:
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
            ngram_top_k=ngram_top_k,
            ngram_range=ngram_range,
        )

        # Create model instance.
        model = self._get_mlp_model(
            mlp_dense_units=mlp_dense_units,
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
        optimizer = Adam(learning_rate=adam_learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=["acc", "accuracy"])

        # Add early stopping
        early_stopping = EarlyStopping(
            monitor="loss", patience=early_stopping_patience, restore_best_weights=True
        )

        # Implement k-fold cross-validation using TensorFlow
        dataset = data.Dataset.from_tensor_slices((x_train, self.train_labels))
        dataset = dataset.shuffle(buffer_size=len(x_train))

        # Split into k folds
        fold_size = len(x_train) // kfolds_n_splits
        fold_histories = []

        for fold in range(kfolds_n_splits):
            if fit_log_verbosity > 0:
                print(f"Fold {fold + 1}/{kfolds_n_splits}")

            # Get indices for train and validation
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            val_data = dataset.skip(val_start).take(fold_size)
            train_data = dataset.take(val_start).concatenate(dataset.skip(val_end))

            # Batch the datasets
            val_data = val_data.batch(fit_batch_size)
            train_data = train_data.batch(fit_batch_size)

            history = model.fit(
                train_data,
                epochs=fit_epochs,
                verbose=fit_log_verbosity,
                callbacks=[early_stopping],
                validation_data=val_data,
            )

            fold_histories.append(history.history)

        # Average the metrics across folds
        avg_metrics = {
            "training_loss": np.mean([h["loss"][-1] for h in fold_histories]),
            "training_accuracy": np.mean([h["acc"][-1] for h in fold_histories]),
            "validation_loss": np.mean([h["val_loss"][-1] for h in fold_histories]),
            "validation_accuracy": np.mean([h["val_acc"][-1] for h in fold_histories]),
        }

        # Get predictions for test data
        predictions = model.predict(x_val)

        return model, predictions, avg_metrics
