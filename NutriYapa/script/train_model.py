import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_tree_model import DecisionTreeWrapper
from src.feature_engineering import compute_nutrient_features
import joblib

# Use absolute paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "recipes.csv")
df = compute_nutrient_features(df)

X = df[['protein_per_serving','calories','fat','carbs']]
y = (df["protein_per_serving"] >= 20).astype(int)

model = DecisionTreeWrapper()
model.train(X, y)
joblib.dump(model.model, MODEL_DIR / "tree.joblib")
print(f"Modelo guardado en {MODEL_DIR / 'tree.joblib'}")
