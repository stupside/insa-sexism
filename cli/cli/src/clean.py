from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, LancasterStemmer


import re

import emoji

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

    # Remove mentions but keep the text
    text = re.sub(r"@", "", text)

    # Remove hashtags but keep the text
    text = re.sub(r"#", "", text)

    text = text.lower()

    # Convert to lowercase and tokenize
    tokens = word_tokenize(text)

    # Remove numbers
    tokens = [word for word in tokens if not word.isnumeric()]

    # Remove punctuations
    tokens = [word for word in tokens if word.isalnum()]

    # Remove short words
    tokens = [word for word in tokens if len(word) > 2]

    # Stem and lemmatize
    filtered = [stemmer.stem(word) for word in tokens]
    filtered = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove stopwords
    filtered = [word for word in filtered if word not in stop_words]

    text = " ".join(filtered)

    return text
