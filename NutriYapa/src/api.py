from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.recommender import NutriRecommender


app = FastAPI()


class UserProfile(BaseModel):
    user_id: str
    lat: float
    lon: float
    goal: str  # 'lose_weight', 'gain_muscle', 'wellness'
    allergies: list = []
    dislikes: list = []


recommender = NutriRecommender()


@app.post('/recommend')
def recommend(profile: UserProfile):
    items = pd.read_csv('data/processed/items_for_api.csv')
    user_profile = profile.dict()
    results = recommender.recommend_for_user(user_profile, items, top_k=10)
    return results.to_dict(orient='records')