import torch

from torch.utils.data import Dataset


from cli.src.clean import clean_text

from cli.src.types.train import TrainData

from cli.src.vectorizer import TextVectorizer

from cli.src.model import device


class TrainDataSet(Dataset):
    datas: list[TrainData] = []

    vectorizer: TextVectorizer

    def __init__(self, vectorizer: TextVectorizer):
        super().__init__()
        self.vectorizer = vectorizer

    def __len__(self):
        return self.datas.__len__()

    def __getitem__(self, idx: int):
        data = self.datas[idx]

        tweet = clean_text(data.tweet)
        sexist = self.check_is_sexist(data)

        tweet = self.vectorizer.vectorize(tweet)
        sexist = torch.tensor(sexist).to(torch.long).to(device)

        return tweet, sexist

    def check_is_sexist(self, data: TrainData) -> bool:
        sexism_votes = [
            1 if label.upper() == "YES" else 0 for label in data.labels_task1
        ]

        if not sexism_votes:  # Handle empty labels
            return False

        return (sum(sexism_votes) / len(data.labels_task1)) >= 0.5
