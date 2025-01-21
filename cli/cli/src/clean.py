from nltk import download
from nltk.corpus import words, stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.stem import WordNetLemmatizer
import re
import emoji
from multiprocessing import Lock
import os

# Global initialization flag and lock
_nltk_initialized = False
_init_lock = Lock()


def initialize_nltk():
    """Initialize NLTK resources once at startup"""
    global _nltk_initialized

    # Use lock to prevent multiple processes from initializing simultaneously
    with _init_lock:
        if not _nltk_initialized:
            if "NLTK_INITIALIZED" not in os.environ:
                # Only download in main process
                download("words")
                download("stopwords")
                download("punkt")
                download("wordnet")
                os.environ["NLTK_INITIALIZED"] = "1"
            _nltk_initialized = True


# Initialize in main process
initialize_nltk()


class Resources:
    def __init__(
        self,
        correct_words: list[str],
        stop_words: list[str],
        lemmatizer: WordNetLemmatizer,
        word_dict: dict[str, list[str]],
    ):
        self.correct_words = correct_words
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        self.word_dict = word_dict


# Lazy loading of resources
def get_resources():
    """Get NLTK resources in a lazy and process-safe way"""
    initialize_nltk()  # Will only download if needed

    resources = Resources(
        correct_words=words.words(),
        stop_words=set(stopwords.words("english")),
        lemmatizer=WordNetLemmatizer(),
        word_dict=_build_word_dict(words.words()),
    )

    return resources


def _build_word_dict(correct_words):
    word_dict: dict[str, list[str]] = {}
    for w in correct_words:
        first_letter = w[0] if w else ""
        if first_letter not in word_dict:
            word_dict[first_letter] = []
        word_dict[first_letter].append(w)
    return word_dict


# Cache for resources in each process
_process_resources: Resources = None


def get_process_resources():
    """Get or initialize resources for current process"""
    global _process_resources
    if _process_resources is None:
        _process_resources = get_resources()
    return _process_resources


def remove_hours(text: str) -> str:
    # Regex pattern to match various time formats
    time_patterns = [
        r"\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?",  # HH:MM or HH:MM AM/PM
        r"\b\d{1,2}\s?(?:AM|PM|am|pm)\b",  # HH AM/PM
        r"\b\d{1,2}\s?o\'?clock\b",  # HH o'clock
    ]

    # Combine patterns and compile regex
    time_regex = re.compile("|".join(time_patterns))

    # Remove all matching time expressions
    cleaned_text = time_regex.sub("", text)

    # Return cleaned text
    return " ".join(cleaned_text.split())  # Remove extra spaces


def correct_word(word: str) -> str:
    if not word:
        return word

    resources = get_process_resources()

    first_letter = word[0]
    if first_letter not in resources.word_dict:
        return word

    # Only consider words with similar length (±1 character)
    word_len = len(word)
    candidates = [
        w for w in resources.word_dict[first_letter] if abs(len(w) - word_len) <= 1
    ]

    # Only compute edit distance for the first 50 candidates
    candidates = candidates[:50]

    try:
        distances = [(edit_distance(word, w), w) for w in candidates]
        min_distance, best_word = min(distances, key=lambda x: x[0])
        # Only correct if edit distance is 1
        return best_word if min_distance == 1 else word
    except ValueError:
        return word


def clean_text(text: str):
    resources = get_process_resources()

    # Convert emojis to text
    text = emoji.demojize(text)

    # Remove mentions and hashtags
    text = re.sub(r"[@#](\w+)", "<MENTION>", text)

    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "<HTTPURL>", text, flags=re.MULTILINE)

    # Remove hours
    text = remove_hours(text)

    # Remove virgulas
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Remove extra spaces
    text = " ".join(text.split())

    # Convert to lowercase and tokenize
    tokens = word_tokenize(text)

    # Remove numbers
    tokens = [word for word in tokens if not word.isnumeric()]

    # Correct words
    tokens = [correct_word(word) for word in tokens]

    # Remove punctuations
    tokens = [word for word in tokens if word.isalnum()]

    # Stem and lemmatize - do not stem as it is too aggressive
    tokens = [resources.lemmatizer.lemmatize(word) for word in tokens]

    # Remove stopwords
    tokens = [word for word in tokens if word not in resources.stop_words]

    # # Remove short words
    # tokens = [word for word in tokens if len(word) > 2]

    text = " ".join(tokens)

    return text
