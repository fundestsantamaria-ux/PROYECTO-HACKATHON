from src.decision_tree_model import DecisionTreeWrapper, DecisionTreeHeuristic
from src.feature_engineering import compute_nutrient_features, identify_ecuadorian_recipes
from src.intelligent_scorer import IntelligentScorer
import pandas as pd
import numpy as np


class NutriRecommender:
    """
    Sistema de recomendaciones nutricionales inteligente
    Integra clasificación heurística, scoring avanzado y filtros personalizados
    """
    
    def __init__(self, model=None):
        self.model = model or DecisionTreeHeuristic()
        self.scorer = IntelligentScorer()

    def recommend_for_user(self, user_profile, items_df, top_k=10, precomputed=False):
        """
        Generar recomendaciones personalizadas para un usuario
        
        Args:
            user_profile: Dict o UserProfile con información del usuario
            items_df: DataFrame con recetas/productos
            top_k: Número de recomendaciones a retornar
            precomputed: Si True, asume que features ya están calculadas
            
        Returns:
            DataFrame con top_k recomendaciones ordenadas por score
        """
        # Convertir UserProfile a dict si es necesario
        if hasattr(user_profile, 'to_dict'):
            user_profile = user_profile.to_dict()
        
        # Copiar datos (shallow copy más rápida)
        items = items_df.copy(deep=False)
        
        # Calcular features solo si no están pre-calculadas
        if not precomputed:
            # Calcular características nutricionales avanzadas
            items = compute_nutrient_features(items)
            
            # Identificar recetas ecuatorianas
            items = identify_ecuadorian_recipes(items)
        
        # Filtrar por alergias (obligatorio) - optimizado
        items = self._filter_allergies(items, user_profile.get('allergies', []))
        
        # Filtrar por gustos (obligatorio) - optimizado
        items = self._filter_dislikes(items, user_profile.get('dislikes', []))
        
        # Filtrar por condiciones de salud (obligatorio)
        items = self._filter_health_conditions(items, user_profile)
        
        if len(items) == 0:
            return items  # No hay items disponibles después de filtrar
        
        # Aplicar modelo de clasificación - vectorizado
        goal = user_profile.get('goal', 'wellness')
        items['score_tag'] = self._batch_classify(items, goal)
        
        # Calcular score inteligente personalizado - vectorizado
        items['score'] = self._batch_score(items, user_profile)
        
        # Bonus adicional por preferencia ecuatoriana
        if user_profile.get('ecuadorian_preference', True):
            items.loc[items['is_ecuadorian'] == True, 'score'] *= 1.15
        
        # Ordenar y retornar top K
        return items.nlargest(top_k, 'score')

    def _filter_allergies(self, items, allergies):
        """Filtrar items que contengan alérgenos (optimizado con vectorización)"""
        if not allergies or 'ingredients' not in items.columns:
            return items
        
        # Convertir alergias a minúsculas una sola vez
        allergies_lower = [a.lower() for a in allergies]
        
        # Mapeo de traducción español-inglés para alergias comunes
        allergy_map = {
            'maní': 'peanut',
            'lácteos': 'dairy',
            'soya': 'soy',
            'trigo': 'wheat',
            'mariscos': 'shellfish',
            'pescado': 'fish',
            'frutos secos': 'tree nuts'
        }
        
        # Agregar versiones en inglés
        for esp, eng in allergy_map.items():
            if esp in allergies_lower and eng not in allergies_lower:
                allergies_lower.append(eng)
        
        # Vectorizar la búsqueda usando str.contains
        ingredients_lower = items['ingredients'].fillna('').astype(str).str.lower()
        
        # Crear patrón regex más eficiente
        pattern = '|'.join(allergies_lower)
        has_allergen = ingredients_lower.str.contains(pattern, na=False, regex=True)
        
        return items[~has_allergen]
    
    def _filter_dislikes(self, items, dislikes):
        """Filtrar items que contengan ingredientes no deseados (optimizado)"""
        if not dislikes or 'ingredients' not in items.columns:
            return items
        
        # Convertir dislikes a minúsculas una sola vez
        dislikes_lower = [d.lower() for d in dislikes]
        
        # Mapeo de traducción español-inglés
        dislike_map = {
            'ajo': 'garlic',
            'cebolla': 'onion',
            'champiñones': 'mushroom',
            'aceitunas': 'olive',
            'pepinillos': 'pickle',
            'mayonesa': 'mayo'
        }
        
        # Agregar versiones en inglés
        for esp, eng in dislike_map.items():
            if esp in dislikes_lower and eng not in dislikes_lower:
                dislikes_lower.append(eng)
        
        # Vectorizar la búsqueda
        ingredients_lower = items['ingredients'].fillna('').astype(str).str.lower()
        pattern = '|'.join(dislikes_lower)
        has_dislike = ingredients_lower.str.contains(pattern, na=False, regex=True)
        
        return items[~has_dislike]
    
    def _batch_classify(self, items, goal):
        """Clasificación por lotes más rápida"""
        # Simplificar clasificación basada en objetivo y macros
        calories = items['calories'].fillna(0)
        protein = items['protein_per_serving'].fillna(0)
        
        if goal == 'lose_weight':
            # Peso: bajo cal, alta proteína
            return pd.cut(
                calories - protein * 10,
                bins=[-np.inf, 200, 400, np.inf],
                labels=['optimal_weightloss', 'lowcal_highprot', 'moderate_weightloss']
            )
        elif goal == 'gain_muscle':
            # Músculo: alta proteína
            return pd.cut(
                protein,
                bins=[0, 20, 30, np.inf],
                labels=['balanced_muscle', 'high_protein', 'optimal_muscle_gain']
            )
        else:
            # Bienestar: balanceado
            return 'balanced_healthy'
    
    def _batch_score(self, items, user_profile):
        """Scoring vectorizado para mejor rendimiento"""
        # Score base por categoría
        category_map = {
            'optimal_weightloss': 100, 'optimal_muscle_gain': 100, 'optimal_wellness': 100,
            'very_high_protein': 95, 'very_low_cal_protein': 95, 'nutrient_dense': 95,
            'lowcal_highprot': 90, 'high_protein_bulk': 90, 'balanced_healthy': 90,
            'high_protein': 80, 'balanced_muscle': 70, 'moderate_weightloss': 65,
            'balanced': 60, 'standard': 50
        }
        
        # Convertir score_tag a string para evitar problemas con Categorical
        score_tag_str = items['score_tag'].astype(str)
        score = score_tag_str.map(category_map).fillna(50).astype(float)
        
        # Bonus por densidad nutricional (si existe)
        if 'nutrient_density' in items.columns:
            score += items['nutrient_density'].fillna(0) * 10
        
        # Bonus por proteína
        protein_bonus = np.clip(items['protein_per_serving'].fillna(0) / 2, 0, 20)
        score += protein_bonus
        
        # Penalización por calorías extremas según objetivo
        goal = user_profile.get('goal', 'wellness')
        if goal == 'lose_weight':
            score -= np.clip((items['calories'].fillna(0) - 400) / 50, 0, 20)
        elif goal == 'gain_muscle':
            score -= np.clip((300 - items['calories'].fillna(0)) / 50, 0, 20)
        
        return np.clip(score, 0, 150)
    
    def _filter_health_conditions(self, items, user_profile):
        """
        Filtrar recetas incompatibles con condiciones de salud
        """
        health_conditions = user_profile.get('health_conditions', [])
        
        if not health_conditions:
            return items
        
        def is_safe_for_conditions(row):
            """Verificar si una receta es segura para las condiciones del usuario"""
            for condition in health_conditions:
                if condition == 'diabetes':
                    # Rechazar si tiene demasiado azúcar o carbohidratos
                    if row.get('sugar', 0) > 25:
                        return False
                    if row.get('carbs', 0) > 70:
                        return False
                
                elif condition == 'hypertension':
                    # Rechazar si tiene demasiado sodio
                    if row.get('sodium', 0) > 1000:
                        return False
                
                elif condition == 'high_cholesterol':
                    # Rechazar si tiene demasiada grasa saturada
                    if row.get('saturated_fat', 0) > 20:
                        return False
                
                elif condition == 'kidney_disease':
                    # Rechazar si tiene demasiada proteína o sodio
                    if row.get('protein_per_serving', 0) > 35:
                        return False
                    if row.get('sodium', 0) > 800:
                        return False
                
                elif condition == 'celiac':
                    # Verificar ingredientes sin gluten
                    ingredients = str(row.get('ingredients', '')).lower()
                    if any(grain in ingredients for grain in 
                          ['wheat', 'trigo', 'barley', 'cebada', 'rye', 'centeno']):
                        return False
            
            return True
        
        return items[items.apply(is_safe_for_conditions, axis=1)]

    def get_meal_plan(self, user_profile, items_df, days=1):
        """
        Generar plan de comidas para varios días (optimizado)
        
        Args:
            user_profile: Perfil del usuario
            items_df: DataFrame con recetas disponibles
            days: Número de días a planificar
            
        Returns:
            Dict con plan de comidas organizado por día y tipo de comida
        """
        meal_plan = {}
        
        # Convertir UserProfile a dict si es necesario
        if hasattr(user_profile, 'to_dict'):
            profile_dict = user_profile.to_dict()
        else:
            profile_dict = user_profile
        
        target_calories = profile_dict.get('target_calories', 2000)
        
        # PRE-CALCULAR features UNA SOLA VEZ para todas las recetas
        items_processed = items_df.copy()
        items_processed = compute_nutrient_features(items_processed)
        items_processed = identify_ecuadorian_recipes(items_processed)
        
        # Distribución de calorías por comida (estilo ecuatoriano)
        meal_distributions = {
            'breakfast': 0.25,  # Desayuno 25%
            'lunch': 0.40,      # Almuerzo 40% (comida principal)
            'dinner': 0.25,     # Merienda 25%
            'snack': 0.10       # Snack 10%
        }
        
        # Crear índices por rango calórico para búsqueda rápida
        calorie_ranges = {}
        for meal_type, calorie_pct in meal_distributions.items():
            meal_target = target_calories * calorie_pct
            mask = (
                (items_processed['calories'] >= meal_target * 0.7) &
                (items_processed['calories'] <= meal_target * 1.3)
            )
            calorie_ranges[meal_type] = items_processed[mask].copy()
        
        for day in range(1, days + 1):
            daily_plan = {}
            used_recipes = set()
            
            for meal_type, calorie_pct in meal_distributions.items():
                # Obtener recetas pre-filtradas para este rango calórico
                suitable_items = calorie_ranges[meal_type]
                
                # Excluir recetas ya usadas este día
                suitable_items = suitable_items[
                    ~suitable_items['name'].isin(used_recipes)
                ]
                
                if len(suitable_items) > 0:
                    # Obtener mejor recomendación (usar features pre-calculadas)
                    recommendations = self.recommend_for_user(
                        profile_dict,
                        suitable_items,
                        top_k=1,
                        precomputed=True
                    )
                    
                    if len(recommendations) > 0:
                        recipe = recommendations.iloc[0]
                        daily_plan[meal_type] = {
                            'name': recipe['name'],
                            'calories': recipe['calories'],
                            'protein': recipe['protein_per_serving'],
                            'score': recipe.get('score', 0),
                            'category': recipe.get('score_tag', 'standard')
                        }
                        used_recipes.add(recipe['name'])
            
            meal_plan[f'day_{day}'] = daily_plan
        
        return meal_plan