# Clash Predict

A machine learning pipeline for predicting Clash Royale 1v1 ladder battle outcomes based on deck composition, card levels, and trophy counts.

## Setup

### Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna joblib requests beautifulsoup4 python-dotenv
```

Or install the package directly:

```bash
pip install -e .
```

### API Token

Get a Clash Royale API token from the [Supercell Developer Portal](https://developer.clashroyale.com). Create a `.env.local` file in the project root:

```
CLASH_ROYALE_API_TOKEN=your_token_here
```

## Usage

### 1. Collect Data

Use `data_collection/clashapi.py` to gather battle data from the Clash Royale API. The collector uses BFS to crawl player battle logs, starting from seed clans/players.

Edit the seed clans and players at the bottom of the file, then run:

```bash
python -m data_collection.clashapi
```

This saves raw battle records to `data/raw/battles.jsonl`. The collector automatically saves state so runs can be resumed.

### 2. Preprocess Data

Transform raw battle JSON into a feature matrix (CSV) with normalized card levels, support card features, deck balance features, trophy diff, and average elixir cost:

```bash
python -m preprocessing.preprocessing
```

Output: `data/processed/battles_cards.csv`

### 3. Train Models

**Compare models** across Logistic Regression, Random Forest, LightGBM, and XGBoost with cross-validation:

```bash
python -m models.compare_models
```

**Train the main XGBoost model** with Optuna hyperparameter tuning (50 trials, 3-fold CV):

```bash
python -m models.train_xgboost
```

Output: `models/xgb_clash_model.joblib`

### 4. Run Predictions

Use the prediction interface to estimate win probability for a given matchup:

```python
from inference.predict import predict_win_prob

team_cards = [
    {"name": "Hog Rider", "elixirCost": 4, "level": 11, "maxLevel": 11},
    # ... 7 more cards
]
opp_cards = [
    {"name": "Golem", "elixirCost": 8, "level": 11, "maxLevel": 11},
    # ... 7 more cards
]

loss_prob, win_prob = predict_win_prob(
    team_trophies=5500,
    opp_trophies=5800,
    team_cards=team_cards,
    opp_cards=opp_cards,
)
print(f"Win probability: {win_prob:.2%}")
```

### 5. Exploratory Data Analysis

Open the Jupyter notebook for visualizations and analysis of battle patterns, card usage, and win rates:

```bash
jupyter notebook eda/eda.ipynb
```
