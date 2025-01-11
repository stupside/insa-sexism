from tensorflow import cast, float32

from numpy import array, append, ndarray

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
        # This is used to standardize, tokenize, and vectorize our text data
        vectorizer = TextVectorization(
            max_tokens=ngram_top_k,
            output_mode="count",
            ngrams=ngram_range,
            output_sequence_length=None,
            standardize="lower_and_strip_punctuation",
        )

        # Adapt the vectorizer to the training data
        vectorizer.adapt(self.train_tweets)

        # Transform the texts to vectors using eager execution
        x_val = vectorizer(self.test_tweets)
        x_train = vectorizer(self.train_tweets)

        # Convert to numpy arrays using tf.cast first
        x_val: ndarray = cast(x_val, float32)._numpy()
        x_train: ndarray = cast(x_train, float32)._numpy()

        return x_train, x_val

    @staticmethod
    def _get_last_layer_units_and_activation(num_classes: int):
        # Determine the number of units and activation function for the last layer
        if num_classes == 2:  # Binary classification
            activation = "sigmoid"
            units = 1
        else:  # Multi-class classification
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
        # Get the number of units and the activation function for the last layer
        op_units, op_activation = self._get_last_layer_units_and_activation(num_classes)

        inputs = Input(shape=input_shape)

        # Add BatchNormalization at the start
        # This normalizes the input to the next layer
        chain = inputs
        chain = BatchNormalization()(chain)

        # Add hidden layers
        for i in range(layers):
            # Decrease units gradually in deeper layers
            layer_units = mlp_dense_units // (2**i)
            if layer_units < 32:  # Minimum number of units
                layer_units = 32

            # Add Dense layer with L2 regularization
            # This helps prevent overfitting and improves generalization of the model
            chain = Dense(
                units=layer_units,
                activation="relu",  # Rectified Linear Unit activation function
                kernel_regularizer=l2(0.01),  # L2 regularization
            )(chain)

            # Add BatchNormalization
            # This normalizes the input to the next layer
            chain = BatchNormalization()(chain)

            # Add Dropout
            # This helps prevent overfitting by randomly setting a fraction of input units to 0
            # Nombre de neuronne activer a chaque epoch.
            # Add more dropout layers for more regularization and less overfitting
            chain = Dropout(rate=dropout_rate)(chain)

        # Add the last layer
        # This is the output layer, which is used to make predictions
        outputs = Dense(units=op_units, activation=op_activation)(chain)

        # Create the model
        # This is the final model that will be trained
        model = models.Model(inputs=inputs, outputs=outputs)

        # Return the model
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

        # Vectorize the data using n-grams
        x_train, x_val = self._ngram_vectorize(
            ngram_top_k=ngram_top_k,
            ngram_range=ngram_range,
        )

        # Get number of classes from training data only
        num_classes = self._get_num_classes()

        # Create the model using the vectorized data
        # This model will be trained on the data to make predictions later
        model = self._get_mlp_model(
            mlp_dense_units=mlp_dense_units,
            layers=layers,
            input_shape=x_train.shape[1:],
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Determine the loss function based on the number of classes
        # This is used to measure how well the model is performing during training
        if num_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"

        model = self._get_mlp_model(
            mlp_dense_units=mlp_dense_units,
            layers=layers,
            input_shape=x_train.shape[1:],
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Adam is used to minimize the loss function during training.
        # It compare to SGD (Stochastic Gradient Descent), but is more efficient and requires less memory.
        # It is a function or an algorithm that adjusts the attributes of the NN, such as weights and learning rates.
        fold_optimizer = Adam(learning_rate=adam_learning_rate)

        model.compile(optimizer=fold_optimizer, loss=loss, metrics=["accuracy"])

        # Initialize stratified k-fold and callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )

        history = model.fit(
            x=x_train,
            y=self.train_labels,
            validation_data=(x_train, self.train_labels),
            epochs=fit_epochs,
            batch_size=fit_batch_size,
            callbacks=[early_stopping],
            verbose=fit_log_verbosity,
        )

        history = history.history

        # Calculate average metrics
        avg_metrics = {
            "accuracy": np.mean(history["accuracy"]),
            "loss": np.mean(history["loss"]),
            "val_accuracy": np.mean(history["val_accuracy"]),
            "val_loss": np.mean(history["val_loss"]),
        }

        # Use best model for predictions
        predictions = model.predict(x_val)

        return model, predictions, avg_metrics
