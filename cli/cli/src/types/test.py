from typing import Literal

type AnnotatorGender = Literal["F", "M"]


class TestData:
    ID: int

    tweet: str

    # Annotation
    annotators: list[str]
    age_annotators: list[str]
    gender_annotators: list[AnnotatorGender]
    countries_annotators: list[str]
    ethnicities_annotators: list[str]
    study_levels_annotators: list[str]

    def __init__(self, **fields):
        self.__dict__.update(**fields)
