class MLP_MODEL_PARAMS_WRAPPER:

    num_classes: int = 2

    layers: int = 2

    units: int = 4

    dropout_rate: float = 2

    input_shape: tuple[int, int] = (1, 5)

    def __init__(self, **fields):

        self.__dict__.update(**fields)


class MLP_TWEETS_LABELS_WRAPPER:
    labels: list[int] = []
    tweets: list[str] = []


class MLP_PREPOCESS_PARAM_WRAPPER:

    dtype: float

    min_df: float = 1

    analyzer: str = "word"

    ngram_range: tuple[int, int] = (1, 2)

    decode_error: str = "strict"

    strip_accents: str = "unicode"

    lowercase: bool = True

    def __init__(self, **fields):
        self.__dict__.update(**fields)
