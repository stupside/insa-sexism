from typing import Literal

type LabelTask1 = Literal["YES", "NO"]
type LabelTask2 = Literal["DIRECT", "JUDGEMENTAL", "REPORTED", "-"]

type AnnotatorGender = Literal["F", "M"]


class TrainData:
    ID: int

    tweet: str

    # Annotation
    annotators: list[str]
    age_annotators: list[str]
    gender_annotators: list[AnnotatorGender]
    countries_annotators: list[str]
    ethnicities_annotators: list[str]
    study_levels_annotators: list[str]

    # Labels
    labels_task1: list[LabelTask1]  # YES or NO
    labels_task2: list[LabelTask2]  # DIRECT, JUDGEMENTAL, REPORTED, -

    def __init__(self, **fields):
        self.__dict__.update(**fields)
