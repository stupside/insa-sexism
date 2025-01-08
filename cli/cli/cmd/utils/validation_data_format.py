class ValidationDataFormat:

    ID: int

    tweet: str

    annotators: list[str]

    gender_annotators: list[str]

    age_annotators: list[str]

    ethnicities_annotators: list[str]

    study_levels_annotators: list[str]

    countries_annotators: list[str]

    def __init__(self, **fields):

        self.__dict__.update(**fields)
