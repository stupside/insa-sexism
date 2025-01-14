from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer


import re

import emoji

download("wordnet")
download("stopwords")

stop_words = set(stopwords.words("english"))

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


@staticmethod
def clean_text(text: str):

    # Convert emojis to text
    text = emoji.demojize(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r"[@#](\w+)", "", text)

    # Lowercase the text
    text = text.lower()

    # Remove elongated words (e.g., 'loooove' -> 'love')
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Convert to lowercase and tokenize
    tokens = word_tokenize(text)

    # Remove numbers
    tokens = [word for word in tokens if not word.isnumeric()]

    # Remove punctuations
    tokens = [word for word in tokens if word.isalnum()]

    # Stem and lemmatize
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # # Remove short words
    # tokens = [word for word in tokens if len(word) > 2]

    text = " ".join(tokens)

    return text
