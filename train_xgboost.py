# train_xgboost.py
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier

DATA_PATH = Path("processed/battles_cards.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_clash_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop non-feature columns
    drop_cols = ["label", "player_tag", "battle_time", "mode"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])  # Only numeric columns

    y = df["label"]

    print("Feature columns:", list(X.columns))
    print("Label distribution:\n", y.value_counts())

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # Initialize an XGBoost classifier
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # Train
    xgb_clf.fit(X_train, y_train)

    # Evaluation
    y_pred = xgb_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred))

    return xgb_clf


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run preprocessing first.")

    X, y = load_data()
    clf = train_model(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved XGBoost model to {MODEL_PATH}")

    # Example prediction
    sample = X.iloc[[0]]
    proba = clf.predict_proba(sample)[0]
    print("\nExample prediction for first row:")
    print("P(loss) =", proba[0], "P(win) =", proba[1])


if __name__ == "__main__":
    main()