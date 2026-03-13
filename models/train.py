# train.py
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = Path("data/processed/battles_cards.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "clash_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Columns we do NOT want as features
    drop_cols = ["label", "player_tag", "battle_time", "mode"]

    # Keep only numeric feature columns
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])  # ensure all features are numeric

    y = df["label"]

    print("Feature columns:", list(X.columns))
    print("Label distribution:\n", y.value_counts())

    return X, y


def train_model(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # Random Forest works well out of the box for tabular stuff
    clf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=800,
        max_depth=8,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred))

    return clf


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run preprocessing/preprocessing.py first.")

    X, y = load_data()
    clf = train_model(X, y)

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # Example: show predicted win probability for first row
    sample = X.iloc[[0]]  # double brackets -> keep as DataFrame
    proba = clf.predict_proba(sample)[0]
    print("\nExample prediction for first row:")
    print(sample)
    print("P(loss) =", proba[0], " P(win) =", proba[1])


if __name__ == "__main__":
    main()
