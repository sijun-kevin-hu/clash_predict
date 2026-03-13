# train_xgboost.py
import optuna
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/battles_cards.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_clash_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop non-feature columns
    drop_cols = ["label", "player_tag", "battle_time", "mode", "team_trophies", "opp_trophies"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])  # Only numeric columns

    y = df["label"]

    print("Feature columns:", list(X.columns))
    print("Label distribution:\n", y.value_counts())

    return X, y

def make_objective(X_train, y_train):
    def objective(trial):
        # Optuna suggests hyperparameter values
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
        }
        
        model = XGBClassifier(**params)
        
        score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
        return score
    return objective

def train_model(X_train, X_test, y_train, y_test, best_params):
    # Initialize an XGBoost classifier
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **best_params
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
    # Check path
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH}. Run preprocessing/preprocessing.py first.")

    # Load and split test and train
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )
    
    # Run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(X_train, y_train), n_trials=50)
    
    # Print Results
    print("Best params: ", study.best_params)
    print("Best CV accuracy: ", study.best_value)
    
    # Pass best params into train_model
    clf = train_model(X_train, X_test, y_train, y_test, best_params=study.best_params)

    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved XGBoost model to {MODEL_PATH}")

    # Example prediction
    sample = X.iloc[[0]]
    proba = clf.predict_proba(sample)[0]
    print("\nExample prediction for first row:")
    print("P(loss) =", proba[0], "P(win) =", proba[1])


if __name__ == "__main__":
    main()
