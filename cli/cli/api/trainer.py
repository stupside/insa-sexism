from numpy import float64, float32, array, append, ndarray

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from keras._tf_keras.keras import models, Input
from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2

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
        ngram_min_df: int,
        ngram_token_mode: str,
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
            "min_df": ngram_min_df,
            "analyzer": ngram_token_mode,
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
        selector = SelectKBest(f_classif, k=min(ngram_top_k, x_train.shape[1])).fit(
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
        ngram_min_df: int,
        dropout_rate: float,
        fit_batch_size: int,
        mlp_dense_units: int,
        kfolds_n_splits: int,
        ngram_token_mode: str,
        fit_log_verbosity: int,
        adam_learning_rate: float,
        early_stopping_patience: int,
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
            ngram_top_k=ngram_top_k,
            ngram_range=ngram_range,
            ngram_min_df=ngram_min_df,
            ngram_token_mode=ngram_token_mode,
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
        model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

        # Add early stopping
        early_stopping = EarlyStopping(
            monitor="loss", patience=early_stopping_patience, restore_best_weights=True
        )

        # Implement k-fold cross-validation

        kf = KFold(n_splits=kfolds_n_splits, shuffle=True)
        fold_histories = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
            if fit_log_verbosity > 0:
                print(f"Fold {fold + 1}/{kfolds_n_splits}")

            x_train_fold = x_train[train_idx]
            y_train_fold = self.train_labels[train_idx]

            x_val_fold = x_train[val_idx]
            y_val_fold = self.train_labels[val_idx]

            history = model.fit(
                x_train_fold,
                y_train_fold,
                epochs=fit_epochs,
                verbose=fit_log_verbosity,
                callbacks=[early_stopping],
                batch_size=fit_batch_size,
                validation_data=(x_val_fold, y_val_fold),
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
