from torch.utils.data import Dataset

from cli.src.clean import clean_text

from cli.src.types.test import TestData

from cli.src.vectorizer import TextVectorizer


class TestDataSet(Dataset):
    datas: list[TestData] = []

    vectorizer: TextVectorizer

    def __init__(self, vectorizer: TextVectorizer):
        super().__init__()
        self.vectorizer = vectorizer

    def __len__(self):
        return self.datas.__len__()

    def __getitem__(self, idx: int):
        data = self.datas[idx]

        tweet = clean_text(data.tweet)

        return self.vectorizer.vectorize(tweet)
