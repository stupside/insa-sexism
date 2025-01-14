import torch
from collections import Counter
from .clean import clean_text


class TextVectorizer:
    def __init__(self, max_length: int):
        self.vocabulary = {"<PAD>": 0, "<UNK>": 1}
        self.max_length = max_length

    def fit(self, texts: list[str]):
        # Clean and tokenize all texts
        word_counts = Counter()
        for text in texts:
            text = clean_text(text)
            words = text.split()
            word_counts.update(words)

        # Add all unique words to vocabulary
        for word in word_counts:
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)

    def vectorize(self, text: str) -> torch.Tensor:
        text = clean_text(text)
        words = text.split()

        # Convert to indices with truncation and padding
        indices = [
            self.vocabulary.get(word, self.vocabulary["<UNK>"])
            for word in words[: self.max_length]
        ]

        # Pad if necessary
        if len(indices) < self.max_length:
            indices.extend(
                [self.vocabulary["<PAD>"]] * (self.max_length - len(indices))
            )

        return torch.tensor(indices[: self.max_length], dtype=torch.long)
