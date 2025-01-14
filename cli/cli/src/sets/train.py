import torch

from torch.utils.data import Dataset


from cli.src.clean import clean_text

from cli.src.types.train import TrainData

from cli.src.vectorizer import TextVectorizer


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
        sexist = torch.tensor(sexist, dtype=torch.long)

        return tweet, sexist

    def check_is_sexist(self, data: TrainData) -> bool:

        # Initialize weights for different types of sexism
        sexism_weights = {
            "-": 0.5,
            "DIRECT": 1.0,
            "JUDGEMENTAL": 0.8,
            "REPORTED": 0.7,
        }

        # Initialize demographic weights (optional tuning based on analysis)
        demographic_weights = {
            "gender": {"F": 1.2, "M": 0.8},  # Slight bias towards female annotators
        }

        total_score = 0
        valid_votes = 0

        for idx in range(len(data.annotators)):

            task1_label = data.labels_task1[idx]
            task2_label = data.labels_task2[idx]

            gender = data.gender_annotators[idx]

            # Skip invalid or incomplete annotations
            if task1_label not in ["YES", "NO"] or task2_label not in sexism_weights:
                continue

            vote_score = 0
            valid_votes += 1

            # Calculate sexism score
            if task1_label == "YES":
                vote_score = sexism_weights[task2_label]
            elif task1_label == "NO":
                vote_score = -0.5  # Slight penalty for NO votes

            # Apply demographic adjustments (if available)
            if gender in demographic_weights["gender"]:
                vote_score *= demographic_weights["gender"][gender]

            total_score += vote_score

        # Normalize score (-1 to 1 range)
        if valid_votes == 0:
            raise ValueError("No valid votes found for this data point.")

        normalized_score = total_score / valid_votes

        return normalized_score >= 0.0
