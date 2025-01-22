import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
import emoji
from tqdm import tqdm
from functools import lru_cache

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


@lru_cache(maxsize=10000)
def lemmatize_word(word):
    """Cache lemmatization results"""
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


def extract_ngrams(text):
    """Extract n-grams of sizes 1, 2, and 3 from text"""
    tokens = word_tokenize(text)
    unigrams = [" ".join(gram) for gram in ngrams(tokens, 1)]
    bigrams = [" ".join(gram) for gram in ngrams(tokens, 2)]
    trigrams = [" ".join(gram) for gram in ngrams(tokens, 3)]
    return " ".join(unigrams + bigrams + trigrams)


def preprocess_text(text):
    """Sequential preprocessing with progress tracking"""
    # Convert to lowercase and basic cleaning
    text = str(text).lower()

    # Quick replacements for special tokens
    text = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", text)
    text = re.sub(r"@\w+", " <USER> ", text)
    text = re.sub(r"#(\w+)", r" <HASHTAG> \1 ", text)

    # Remove emojis efficiently
    text = emoji.replace_emoji(text, " <EMOJI> ")

    # Clean text while preserving tokens
    text = re.sub(r"[^\w\s<>]", " ", text)

    # Efficient tokenization and lemmatization
    tokens = [
        lemmatize_word(token)
        for token in text.split()
        if token not in stopwords.words("english")
    ]

    return " ".join(tokens)


def process_annotator_data(row):
    """Process only gender and age information"""
    gender_counts = pd.Series(eval(row["gender_annotators"])).value_counts()
    age_counts = pd.Series(eval(row["age_annotators"])).value_counts()

    features = {
        "female_ratio": gender_counts.get("F", 0) / len(eval(row["gender_annotators"])),
        "young_ratio": sum(1 for age in eval(row["age_annotators"]) if age == "18-22")
        / len(eval(row["age_annotators"])),
    }
    return pd.Series(features)


def process_labels(row):
    """Convert annotator labels to final label"""
    # Convert string representation of list to actual list
    labels = eval(row["labels_task1"])
    # Count YES vs NO
    yes_count = labels.count("YES")
    no_count = labels.count("NO")
    # Return majority vote
    return 1 if yes_count > no_count else 0


def get_label_type(row):
    """Get type of sexism from task2 labels"""
    labels = eval(row["labels_task2"])
    # Remove '-' entries
    valid_labels = [l for l in labels if l != "-"]
    if not valid_labels:
        return "NONE"
    # Return most common label
    return max(set(valid_labels), key=valid_labels.count)


def get_annotator_indices(annotators, gender, age):
    """Get indices of annotators matching specified gender and age"""
    indices = []
    for i, (gender_list, age_list) in enumerate(annotators):
        # Convert string lists to actual lists
        genders = eval(gender_list)
        ages = eval(age_list)
        # Check each annotator's gender and age
        for j, (g, a) in enumerate(zip(genders, ages)):
            if g == gender and a == age:
                indices.append(j)
    # Return unique indices
    return list(set(indices))


def get_single_annotator_label(row, annotator_idx):
    """Get label from a single annotator"""
    labels = eval(row["labels_task1"])
    if annotator_idx < len(labels):
        return 1 if labels[annotator_idx] == "YES" else 0
    return None


def get_annotator_demographics(row, annotator_idx):
    """Get only gender and age information"""
    try:
        gender = eval(row["gender_annotators"])[annotator_idx]
        age = eval(row["age_annotators"])[annotator_idx]
        return {
            "gender": gender,
            "age": age,
        }
    except:
        return None


def get_demographic_labels(row, gender, age):
    """Get labels only from annotators of specific gender and age"""
    labels = []
    genders = eval(row["gender_annotators"])
    ages = eval(row["age_annotators"])
    annotations = eval(row["labels_task1"])

    for g, a, label in zip(genders, ages, annotations):
        if g == gender and a == age:
            labels.append(1 if label == "YES" else 0)

    if not labels:
        return None
    # Return majority vote
    return 1 if sum(labels) > len(labels) / 2 else 0


# Load the data
df = pd.read_csv("../../../cli-data/train.csv")

# Create binary labels from task1 annotations
df["label"] = df.apply(process_labels, axis=1)

# Get sexism type labels from task2
df["label_type"] = df.apply(get_label_type, axis=1)

# Add annotator features
annotator_features = df.apply(process_annotator_data, axis=1)
df = pd.concat([df, annotator_features], axis=1)

# Preprocess tweets sequentially with progress bar
print("Preprocessing tweets...")
df["processed_tweets"] = [
    preprocess_text(text) for text in tqdm(df["tweet"], desc="Preprocessing tweets")
]

# Convert labels to numeric values
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # Assuming 'label' is your target column

# Print initial class distribution
print("\nInitial class distribution:")
print(df["label"].value_counts())

# Get annotator combinations for females 18-22
annotator_data = list(zip(df["gender_annotators"], df["age_annotators"]))

# Define demographic groups to analyze
demographic_groups = [
    ("F", "18-22"),  # Young females
    ("F", "23-45"),  # Adult females
    ("F", "46+"),  # Older females
    ("M", "18-22"),  # Young males
    ("M", "23-45"),  # Adult males
    ("M", "46+"),  # Older males
]

# Create results container for all groups
all_groups_results = []

# Create separate models for each group
results_dict = {}
for gender, age in tqdm(demographic_groups, desc="Processing demographic groups"):
    print(f"\n{'='*70}")
    print(f"Training model for {gender} annotators aged {age}")
    print(f"{'='*70}")

    # Get labels from specific demographic group
    df["demographic_label"] = df.apply(
        lambda row: get_demographic_labels(row, gender, age), axis=1
    )

    # Filter out rows without labels from this demographic
    df_demographic = df.dropna(subset=["demographic_label"]).copy()

    print(f"\nFound {len(df_demographic)} samples labeled by {gender} {age}")
    if len(df_demographic) < 100:
        print(f"Skipping {gender} {age} - insufficient data")
        continue

    # Use demographic-specific labels
    X = df_demographic["processed_tweets"]
    y = df_demographic["demographic_label"]

    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts())

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optimize TF-IDF
    tfidf = TfidfVectorizer(
        max_features=3000,  # Reduced features for speed
        stop_words="english",
        ngram_range=(1, 3),
        min_df=3,
        use_idf=True,
        norm="l2",
    )

    print("Fitting TF-IDF...")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

    print("Training SVM...")
    model = SVC(
        kernel="linear",
        C=1.0,
        random_state=42,
        class_weight="balanced",  # Add class weights
    )
    model.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate with proper scaling
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics with proper averaging
    metrics = {
        "gender": gender,
        "age_group": age,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "support": len(y_test),
    }

    # Add validation to results using y.value_counts() instead of label_dist
    metrics.update(
        {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_distribution": dict(y.value_counts()),
        }
    )

    # Store results only if valid
    if metrics["f1"] > 0:  # Basic sanity check
        results_dict[(gender, age)] = metrics
        all_groups_results.append(metrics)

    print(f"\nClass distribution for {gender} {age}:")
    print(y.value_counts())

# Modified summary creation - remove std calculations
final_df = pd.DataFrame(all_groups_results)
print("\nRaw results before averaging:")
print(final_df[["gender", "age_group", "f1", "support", "train_size", "test_size"]])

# Simplified summary without std
weighted_summary = (
    final_df.groupby(["gender", "age_group"])
    .agg(
        {
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "support": "sum",
            "train_size": "sum",
        }
    )
    .round(4)
)

print("\nSummary by Demographic Group:")
print(weighted_summary)

# Save results
weighted_summary.to_csv("../../../cli-data/demographic_analysis_summary.csv")
