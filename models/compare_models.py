# compare_models.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/battles_cards.csv")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,      # default 100 often doesn't converge, 1000 is safe
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,   # 100 trees, good default
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,   # more trees, fast enough on M4
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost (tuned)": XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=407,
    max_depth=8,
    learning_rate=0.07667473794423303,
    subsample=0.8255123715719369,
    colsample_bytree=0.8892336207036892,
    min_child_weight=9,
    random_state=42,
    n_jobs=-1,
    )
}

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
    
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        results[name] = scores.mean()
        print(f"{name}: {scores.mean():.4f}")

    # Print leaderboard
    print("\n--- Leaderboard ---")
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{score:.4f}  {name}")

if __name__ == "__main__":
    main()
