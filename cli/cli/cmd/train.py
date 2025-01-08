# tweet,annotators,gender_annotators,age_annotators,ethnicities_annotators,study_levels_annotators,countries_annotators,labels_task1,labels_task2,ID
class TrainData:

    ID: int
    tweet: str

    # Annotation
    annotators: list[str]
    age_annotators: list[str]
    gender_annotators: list[str]
    countries_annotators: list[str]
    ethnicities_annotators: list[str]
    study_levels_annotators: list[str]

    # Labels
    labels_task1: list[str]
    labels_task2: list[str]

    def __init__(self, **fields):
        self.__dict__.update(**fields)


class TrainSet:

    dataset: list[TrainData]

    def __init__(self):
        self.dataset = []

    def add(self, data: TrainData):
        self.dataset.append(data)


def run(trainset: TrainSet):
    print(f"Loaded {len(trainset.dataset)} rows")
