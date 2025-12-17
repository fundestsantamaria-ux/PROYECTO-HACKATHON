import pandas as pd
from src.config import DATA_DIR


def load_combined_data():
    recipes = pd.read_csv(DATA_DIR / 'recipes.csv')
    products = pd.read_csv(DATA_DIR / 'products.csv')
    venues = pd.read_csv(DATA_DIR / 'venues.csv')
    # join / normalize
    return recipes, products, venues