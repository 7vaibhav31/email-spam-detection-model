import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import TEXT_COL, LABEL_COL, TEST_SIZE, RANDOM_STATE, TFIDF_PARAMS, LR_PARAMS

# Load and prepare data
df = pd.read_csv("../data/email.csv")
df = df[df["Category"].isin(["ham", "spam"])]
df.drop_duplicates(inplace=True)
df["Length"] = df["Message"].apply(len)

x = df[["Message", "Length"]]
y = df["Category"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Define pipeline - DO NOT use fixed TFIDF_PARAMS here
# Let RandomizedSearchCV tune these parameters
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), "Message"),  # Initialize without fixed params
        ("length", StandardScaler(), ["Length"])
    ]
)

pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogisticRegression(random_state=RANDOM_STATE))  # Only set random_state for reproducibility
])

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    # TfidfVectorizer parameters
    "prep__text__max_features": [1000, 2000, 3000, 5000],
    "prep__text__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "prep__text__min_df": [1, 2, 3, 5],
    "prep__text__max_df": [0.7, 0.8, 0.9, 0.95],
    
    # LogisticRegression parameters (using l1_ratio instead of penalty for modern sklearn)
    # Only use saga solver which supports all l1_ratio values (0.0=l2, 0.5=elasticnet, 1.0=l1)
    "model__C": [0.01, 0.1, 1.0, 10, 100],
    "model__l1_ratio": [0.0, 0.5, 1.0],  # 0.0 = l2, 1.0 = l1, 0.5 = elasticnet
    "model__solver": ["saga"],  # saga is the only solver that supports all l1_ratio values
}

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=20,  # Number of combinations to try
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all processors
    random_state=RANDOM_STATE,
    scoring="f1_macro",  # Use macro F1 score for imbalanced classes
    verbose=1
)

# Fit the model
print("Starting RandomizedSearchCV...")
random_search.fit(x_train , y_train)

# Best parameters and score
print("BEST PARAMETERS:", random_search.best_params_)
print("Best F1 Score:", random_search.best_score_)

# Save the best model
joblib.dump(random_search.best_estimator_,"model.pkl")