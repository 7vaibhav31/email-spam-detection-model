
# config.py

DATA_PATH = "../data/email.csv"

TEXT_COL = ["message","Length"]
LABEL_COL = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature flags
ADD_LENGTH_FEATURE = True

# TF-IDF settings
TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "stop_words": "english",
}

# Logistic Regression parameters
LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "liblinear",
    "C": 1.0,
    "class_weight": "balanced",
}
