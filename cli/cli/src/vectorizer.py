import csv
import torch

from collections import Counter

from .model import device


class TextVectorizerOptions:
    max_length: int
    dictionnary_path: str


class TextVectorizer:

    options: TextVectorizerOptions
    vocabulary: dict[str, int]

    def __init__(self, options: TextVectorizerOptions, load: bool = False):
        self.options = options
        if load:  # Load csv file and create vocabulary with csv library
            file = csv.reader(open(self.options.dictionnary_path, "r"))
            self.vocabulary = {word: int(idx) for idx, word in file}
        else:
            self.vocabulary = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts: list[str], save: bool = False):
        # Clean and tokenize all texts
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # Add all unique words to vocabulary
        for word in word_counts:
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)

        if save:
            with open(self.options.dictionnary_path, "w") as file:
                writer = csv.writer(file, lineterminator="\n")
                for word, idx in self.vocabulary.items():
                    writer.writerow([idx, word])

    def vectorize(self, text: str) -> torch.Tensor:
        words = text.split()

        # Convert to indices with truncation and padding
        indices = [
            self.vocabulary.get(word, self.vocabulary["<UNK>"])
            for word in words[: self.options.max_length]
        ]

        # Pad if necessary
        if len(indices) < self.options.max_length:
            indices.extend(
                [self.vocabulary["<PAD>"]] * (self.options.max_length - len(indices))
            )

        return (
            torch.tensor(indices[: self.options.max_length])
            .to(torch.long)
            .to(device=device)
        )
