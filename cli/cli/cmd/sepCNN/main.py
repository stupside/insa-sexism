from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras._tf_keras.keras.preprocessing import text
from keras._tf_keras.keras.preprocessing import sequence
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
import random
import tensorflow as tf

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

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

NUMBER_OF_VOTES = 6


def printHello():
    print("Hello")


def is_sexist(data: list[str]):

    yes = 0

    for label in data:
        if label == "YES":
            yes += 1

    if yes >= (NUMBER_OF_VOTES / 2):

        return 1

    return 0


def prepare_data(brute_training_data=[], testPoportion=0.2, seed=132):

    # Initialize the data
    train_data_neg = []
    train_data_pos = []

    # Extract the tweets and the labels from the training data
    for trainingDataObject in brute_training_data:
        is_sexist_bool = is_sexist(trainingDataObject.labels_task1)
        if is_sexist_bool == 1:
            train_data_pos.append(trainingDataObject.tweet)
        else:
            train_data_neg.append(trainingDataObject.tweet)

    # separate the data into training and testing data based on the testProportion ensuring equal representation of both classes
    train_data = (
        train_data_neg[: int(len(train_data_neg) * (1 - testPoportion))]
        + train_data_pos[: int(len(train_data_pos) * (1 - testPoportion))]
    )
    test_data = (
        train_data_neg[int(len(train_data_neg) * (1 - testPoportion)) :]
        + train_data_pos[int(len(train_data_pos) * (1 - testPoportion)) :]
    )

    # initiate labels for the training and testing data
    train_labels = [0] * len(
        train_data_neg[: int(len(train_data_neg) * (1 - testPoportion))]
    ) + [1] * len(train_data_pos[: int(len(train_data_pos) * (1 - testPoportion))])
    test_labels = [0] * len(
        train_data_neg[int(len(train_data_neg) * (1 - testPoportion)) :]
    ) + [1] * len(train_data_pos[int(len(train_data_pos) * (1 - testPoportion)) :])

    # Shuffle the training data and labels with the same seed so that they are still aligned
    random.seed(seed)
    random.shuffle(train_data)
    random.seed(seed)
    random.shuffle(train_labels)

    # shuffle the test data
    random.seed(seed)
    random.shuffle(test_data)
    random.seed(seed)
    random.shuffle(test_labels)
    return (train_data, train_labels), (test_data, test_labels)


# Vectorizing the data
def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index


# Creating the model
def sepcnn_model(
    blocks,
    filters,
    kernel_size,
    embedding_dim,
    dropout_rate,
    pool_size,
    input_shape,
    num_classes,
    num_features,
    use_pretrained_embedding=False,
    is_embedding_trainable=False,
    embedding_matrix=None,
):
    """Creates an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
        num_features: int, number of words (embedding input dimension).
        use_pretrained_embedding: bool, true if pre-trained embedding is on.
        is_embedding_trainable: bool, true if embedding layer is trainable.
        embedding_matrix: dict, dictionary with embedding coefficients.

    # Returns
        A sepCNN model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()

    # Add embedding layer. If pre-trained embedding is used add weights to the
    # embeddings layer and set trainable to input is_embedding_trainable flag.
    if use_pretrained_embedding:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=embedding_dim,
                input_length=input_shape[0],
                weights=[embedding_matrix],
                trainable=is_embedding_trainable,
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=num_features,
                output_dim=embedding_dim,
                input_length=input_shape[0],
            )
        )

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(
            SeparableConv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                bias_initializer="random_uniform",
                depthwise_initializer="random_uniform",
                padding="same",
            )
        )
        model.add(
            SeparableConv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                bias_initializer="random_uniform",
                depthwise_initializer="random_uniform",
                padding="same",
            )
        )
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(
        SeparableConv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            activation="relu",
            bias_initializer="random_uniform",
            depthwise_initializer="random_uniform",
            padding="same",
        )
    )
    model.add(
        SeparableConv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            activation="relu",
            bias_initializer="random_uniform",
            depthwise_initializer="random_uniform",
            padding="same",
        )
    )
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    return units, activation


# Training the model
FLAGS = None

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000


def train_sequence_model(
    data,
    learning_rate=1e-3,
    epochs=1000,
    batch_size=128,
    blocks=2,
    filters=64,
    dropout_rate=0.2,
    embedding_dim=200,
    kernel_size=3,
    pool_size=3,
):
    """Trains sequence model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        (train_texts, train_labels), (val_texts, val_labels) = data
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of sepCNN layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
        embedding_dim: int, dimension of the embedding vectors.
        kernel_size: int, length of the convolution window.
        pool_size: int, factor by which to downscale input at MaxPooling layer.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError(
            "Unexpected label values found in the validation set:"
            " {unexpected_labels}. Please make sure that the "
            "labels in the validation set are in the same range "
            "as training labels.".format(unexpected_labels=unexpected_labels)
        )

    # Vectorize texts.
    x_train, x_val, word_index = vectorize_data.sequence_vectorize(
        train_texts, val_texts
    )

    # Number of features will be the embedding input dimension. Add 1 for the
    # reserved index 0.
    num_features = min(len(word_index) + 1, TOP_K)

    # Create model instance.
    model = build_model.sepcnn_model(
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        pool_size=pool_size,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
        num_features=num_features,
    )

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
    model.save("rotten_tomatoes_sepcnn_model.h5")
    return history["val_acc"][-1], history["val_loss"][-1]


def main(brute_training_data):
    prepare_data(brute_training_data=brute_training_data)
