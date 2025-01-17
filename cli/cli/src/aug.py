from nltk.corpus import wordnet as wn


def augment(tokens: list[str]) -> list[list[str]]:

    sentences = [tokens]

    for token in tokens:
        synonyms = wn.synonyms(token, lang="eng")

        if synonyms is None:
            continue

        if len(synonyms) == 0:
            continue

        synonyms = synonyms[0]

        for synonym in synonyms:
            sentence = tokens.copy()
            sentence[sentence.index(token)] = synonym[0]

            sentences.append(sentence)

    return sentences
