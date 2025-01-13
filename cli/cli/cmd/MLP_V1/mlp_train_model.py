import tensorflow as tf

# Custom imports
from cli.cmd.utils.training_data_format import TrainingDataFormat
from cli.cmd.utils.validation_data_format import ValidationDataFormat
from cli.cmd.MLP_V1.mlp_preprocess_data import MLP_PREPOCESS_DATA
from cli.cmd.MLP_V1.mlp_prepare_model import MLP_MODEL
from cli.cmd.MLP_V1.wrapper import MLP_PREPOCESS_PARAM_WRAPPER


class MLP_TRAIN_MODEL:

    def __ml_train_model(
        self,
        # DATA
        data,
        training_data: list[TrainingDataFormat],
        validation_data: list[ValidationDataFormat],
        # PARAMS
        vectorizer_param: MLP_PREPOCESS_PARAM_WRAPPER,
        top_k: int,
        learning_rate=1e-3,
        epochs=1000,
        batch_size=128,
        layers=2,
        units=64,
        dropout_rate=0.2,
    ):
        # Get the data.
        (train_texts, train_labels), (val_texts, val_labels) = data

        # Preprocess the data
        data_preprocessor = MLP_PREPOCESS_DATA(
            training_data, validation_data, vectorizer_param, top_k
        )
        data_preprocessor.run()

        # Prepare the model
        model = MLP_MODEL()

        # Compile model with learning parameters.
        if num_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

        # Create callback for early stopping on validation loss. If the loss does
        # not decrease in two consecutive tries, stop training.
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

        # Train and validate model.
        history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size,
        )

        # Print results.
        history = history.history
        print(
            "Validation accuracy: {acc}, loss: {loss}".format(
                acc=history["val_acc"][-1], loss=history["val_loss"][-1]
            )
        )

        # Save model.
        model.save("IMDb_mlp_model.h5")
        return history["val_acc"][-1], history["val_loss"][-1]
