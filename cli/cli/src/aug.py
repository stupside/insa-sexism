import random

from nltk import download
from nltk.corpus import wordnet

download("wordnet")


def augment_with_synonyms(
    tokens: list[str], num_replacements: int, max_augmented: int
) -> list[list[str]]:
    sentences = [tokens]
    replaceable_indices = [
        i for i, token in enumerate(tokens) if wordnet.synsets(token, lang="eng")
    ]

    if not replaceable_indices:
        return sentences  # No words are replaceable

    for _ in range(max_augmented):
        new_sentence = tokens.copy()
        indices_to_replace = random.sample(
            replaceable_indices, min(num_replacements, len(replaceable_indices))
        )

        for idx in indices_to_replace:
            synonyms = wordnet.synsets(tokens[idx], lang="eng")
            if synonyms:
                synonym_lemmas = [
                    lemma.name().replace("_", " ")
                    for syn in synonyms
                    for lemma in syn.lemmas()
                ]
                synonym_lemmas = list(set(synonym_lemmas))  # Remove duplicates
                synonym_lemmas = [
                    syn for syn in synonym_lemmas if syn.lower() != tokens[idx].lower()
                ]  # Exclude the original word

                if synonym_lemmas:
                    new_sentence[idx] = random.choice(synonym_lemmas)

        sentences.append(new_sentence)

    return sentences
