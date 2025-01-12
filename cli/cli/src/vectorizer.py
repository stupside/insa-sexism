import torch

from collections import Counter

from .clean import clean_text


# TODO: https://textattack.readthedocs.io/en/latest/2notebook/3_Augmentations.html
class TextVectorizer:
    def __init__(self, top_k: int):
        self.top_k = top_k
        self.vocabulary = {}

    def fit(self, texts: list[str]):

        # Clean all texts
        texts = [clean_text(text) for text in texts]

        # Count all words across all texts
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # Keep only top_k most common words
        most_common = word_counts.most_common(self.top_k)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

    def vectorize(self, text: str) -> torch.Tensor:
        words = text.split()
        vector = torch.zeros(len(self.vocabulary), dtype=torch.float32)

        # Count words that appear in vocabulary
        word_counts = Counter(words)
        for word, count in word_counts.items():
            if word in self.vocabulary:
                vector[self.vocabulary[word]] = count

        return vector
