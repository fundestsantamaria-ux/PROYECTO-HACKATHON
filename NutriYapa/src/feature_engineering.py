import numpy as np
import pandas as pd


def compute_nutrient_features(df):
    """
    Calcular características nutricionales avanzadas para recetas
    Incluye ratios, densidad nutricional, y métricas de calidad
    """
    # Normalize to per-100g or per-serving
    df['protein_per_serving'] = df['protein'].fillna(0)
    
    # Fill calories with 0 if missing (or use estimated_calories if available)
    if 'estimated_calories' in df.columns:
        df['calories'] = df['calories'].fillna(df['estimated_calories'])
    else:
        df['calories'] = df['calories'].fillna(0)
    
    # Fill other macronutrients
    df['fat'] = df['fat'].fillna(0) if 'fat' in df.columns else 0
    df['carbs'] = df['carbs'].fillna(0) if 'carbs' in df.columns else 0
    df['fiber'] = df['fiber'].fillna(0) if 'fiber' in df.columns else 0
    df['sugar'] = df['sugar'].fillna(0) if 'sugar' in df.columns else 0
    
    # Basic ratios
    df['protein_ratio'] = df['protein_per_serving'] / (df['calories'] + 1)
    df['fat_ratio'] = df['fat'] / (df['calories'] + 1)
    df['carbs_ratio'] = df['carbs'] / (df['calories'] + 1)
    
    # Densidad calórica (kcal por 100g)
    df['calorie_density'] = df['calories'] / 100
    
    # Calidad de carbohidratos (fibra vs azúcar)
    df['carb_quality'] = np.where(
        df['carbs'] > 0,
        (df['fiber'] - df['sugar']) / df['carbs'],
        0
    )
    
    # Score de proteína (g proteína por 100 kcal)
    df['protein_efficiency'] = (df['protein_per_serving'] / (df['calories'] + 1)) * 100
    
    # Densidad nutricional (estimación simple)
    # Más alto = mejor balance de nutrientes por caloría
    df['nutrient_density'] = (
        df['protein_per_serving'] * 2 +  # Proteína es importante
        df['fiber'] * 3 +                 # Fibra muy valorada
        np.maximum(0, 50 - df['sugar'])   # Penalizar azúcar alto
    ) / (df['calories'] + 1)
    
    # Balance de macronutrientes (desviación del ideal 30/30/40)
    protein_cal = df['protein_per_serving'] * 4
    fat_cal = df['fat'] * 9
    carbs_cal = df['carbs'] * 4
    total_macro_cal = protein_cal + fat_cal + carbs_cal + 1
    
    protein_pct = protein_cal / total_macro_cal
    fat_pct = fat_cal / total_macro_cal
    carbs_pct = carbs_cal / total_macro_cal
    
    # Calcular desviación del balance ideal
    df['macro_balance_score'] = 1 - (
        abs(protein_pct - 0.30) +
        abs(fat_pct - 0.30) +
        abs(carbs_pct - 0.40)
    ) / 2
    
    # Clasificación de recetas según características
    df['calorie_category'] = pd.cut(
        df['calories'],
        bins=[0, 200, 400, 600, np.inf],
        labels=['muy_bajo', 'bajo', 'moderado', 'alto']
    )
    
    df['protein_category'] = pd.cut(
        df['protein_per_serving'],
        bins=[0, 10, 20, 30, np.inf],
        labels=['bajo', 'moderado', 'alto', 'muy_alto']
    )
    
    # Índice de saciedad estimado (proteína + fibra, bajo en calorías)
    df['satiety_index'] = (
        df['protein_per_serving'] * 2 +
        df['fiber'] * 3 -
        df['calories'] / 100
    )
    
    # Score de salud general
    df['health_score'] = (
        df['nutrient_density'] * 30 +
        df['protein_efficiency'] * 2 +
        df['carb_quality'] * 20 +
        df['macro_balance_score'] * 30 +
        np.maximum(0, 10 - df['fat']) * 2  # Penalizar grasa excesiva
    )
    
    return df


def identify_ecuadorian_recipes(df):
    """
    Identificar y clasificar recetas ecuatorianas típicas
    
    Args:
        df: DataFrame con columnas 'name' e 'ingredients'
    
    Returns:
        df con columnas adicionales 'is_ecuadorian' y 'ecuadorian_type'
    """
    # Platos típicos ecuatorianos
    ecuadorian_dishes = {
        'costa': [
            'encebollado', 'ceviche', 'corvina', 'encocado', 'viche',
            'bolon', 'tigrillo', 'guatita', 'seco de pollo', 'arroz con menestra'
        ],
        'sierra': [
            'locro', 'fanesca', 'fritada', 'hornado', 'cuy', 'mote',
            'morocho', 'colada morada', 'quimbolitos', 'humitas',
            'llapingachos', 'yahuarlocro'
        ],
        'amazonia': [
            'maito', 'chicha', 'ayampaco'
        ],
        'general': [
            'empanada', 'patacones', 'chifles', 'tostado', 'canguil'
        ]
    }
    
    # Ingredientes típicos ecuatorianos
    ecuadorian_ingredients = [
        'verde', 'platano', 'yuca', 'mote', 'choclo', 'melloco',
        'naranjilla', 'mora', 'taxo', 'maracuya', 'guanabana',
        'achiote', 'aji', 'morocho', 'machica', 'panela'
    ]
    
    df['is_ecuadorian'] = False
    df['ecuadorian_type'] = None
    df['ecuadorian_score'] = 0
    
    if 'name' not in df.columns:
        return df
    
    # Verificar nombres de platos
    for region, dishes in ecuadorian_dishes.items():
        for dish in dishes:
            mask = df['name'].str.lower().str.contains(dish, na=False)
            df.loc[mask, 'is_ecuadorian'] = True
            df.loc[mask, 'ecuadorian_type'] = region
            df.loc[mask, 'ecuadorian_score'] = 100
    
    # Verificar ingredientes si está disponible
    if 'ingredients' in df.columns:
        for ingredient in ecuadorian_ingredients:
            mask = df['ingredients'].str.lower().str.contains(ingredient, na=False)
            df.loc[mask, 'ecuadorian_score'] = df.loc[mask, 'ecuadorian_score'] + 10
    
    # Marcar como ecuatoriana si tiene score alto de ingredientes
    mask_ingredients = df['ecuadorian_score'] >= 20
    df.loc[mask_ingredients & ~df['is_ecuadorian'], 'is_ecuadorian'] = True
    df.loc[mask_ingredients & df['ecuadorian_type'].isna(), 'ecuadorian_type'] = 'general'
    
    return df