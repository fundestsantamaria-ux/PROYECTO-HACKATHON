"""
Sistema de aprendizaje de preferencias de usuario
Aprende de las interacciones del usuario para mejorar recomendaciones
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class PreferenceLearner:
    """
    Sistema de aprendizaje que adapta las recomendaciones basándose en:
    - Recetas seleccionadas por el usuario
    - Recetas rechazadas o ignoradas
    - Patrones de consumo
    - Feedback explícito (ratings)
    """
    
    def __init__(self, storage_path='data/user_preferences'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def record_interaction(self, user_id: str, recipe_id: str, 
                          recipe_features: Dict[str, Any],
                          interaction_type: str, rating: int = None):
        """
        Registrar una interacción del usuario con una receta
        
        Args:
            user_id: ID del usuario
            recipe_id: ID o nombre de la receta
            recipe_features: Características nutricionales de la receta
            interaction_type: 'selected', 'rejected', 'viewed', 'completed'
            rating: Rating opcional del 1-5
        """
        user_file = self.storage_path / f'{user_id}.json'
        
        # Cargar historial existente
        if user_file.exists():
            with open(user_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                'interactions': [],
                'preferences_learned': {},
                'favorite_ingredients': [],
                'disliked_ingredients': []
            }
        
        # Agregar nueva interacción
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'recipe_id': recipe_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'features': recipe_features
        }
        
        history['interactions'].append(interaction)
        
        # Actualizar preferencias aprendidas
        self._update_learned_preferences(history)
        
        # Guardar
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def _update_learned_preferences(self, history: Dict):
        """
        Actualizar preferencias aprendidas basadas en el historial
        """
        interactions = history['interactions']
        
        if len(interactions) < 3:
            return  # No suficientes datos
        
        # Analizar recetas seleccionadas
        selected = [i for i in interactions if i['interaction_type'] == 'selected']
        rejected = [i for i in interactions if i['interaction_type'] == 'rejected']
        rated_high = [i for i in interactions if i.get('rating', 0) >= 4]
        rated_low = [i for i in interactions if i.get('rating', 0) <= 2]
        
        preferences = {}
        
        # Preferencias calóricas
        if selected:
            cal_values = [i['features'].get('calories', 0) for i in selected]
            preferences['preferred_calorie_range'] = {
                'min': np.percentile(cal_values, 25),
                'max': np.percentile(cal_values, 75),
                'mean': np.mean(cal_values)
            }
        
        # Preferencias de proteína
        if selected:
            prot_values = [i['features'].get('protein_per_serving', 0) for i in selected]
            preferences['preferred_protein_range'] = {
                'min': np.percentile(prot_values, 25),
                'max': np.percentile(prot_values, 75),
                'mean': np.mean(prot_values)
            }
        
        # Categorías preferidas
        if selected:
            categories = [i['features'].get('score_tag', '') for i in selected]
            category_counts = pd.Series(categories).value_counts()
            preferences['preferred_categories'] = category_counts.head(5).to_dict()
        
        # Analizar ingredientes (si está disponible)
        fav_ingredients = self._extract_ingredient_patterns(selected)
        disliked_ingredients = self._extract_ingredient_patterns(rejected + rated_low)
        
        history['preferences_learned'] = preferences
        history['favorite_ingredients'] = fav_ingredients
        history['disliked_ingredients'] = disliked_ingredients
    
    def _extract_ingredient_patterns(self, interactions: List[Dict]) -> List[str]:
        """
        Extraer patrones de ingredientes de las interacciones
        """
        all_ingredients = []
        
        for interaction in interactions:
            ingredients_str = interaction['features'].get('ingredients', '')
            if ingredients_str:
                # Simple tokenization
                ingredients = str(ingredients_str).lower().split(',')
                all_ingredients.extend([i.strip() for i in ingredients])
        
        if not all_ingredients:
            return []
        
        # Contar frecuencias
        ingredient_counts = pd.Series(all_ingredients).value_counts()
        
        # Retornar top ingredients
        return ingredient_counts.head(10).index.tolist()
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener preferencias aprendidas del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Dict con preferencias aprendidas
        """
        user_file = self.storage_path / f'{user_id}.json'
        
        if not user_file.exists():
            return {}
        
        with open(user_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        return history.get('preferences_learned', {})
    
    def apply_learned_preferences(self, user_id: str, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Ajustar scores de recomendaciones basándose en preferencias aprendidas
        
        Args:
            user_id: ID del usuario
            recommendations: DataFrame con recomendaciones
            
        Returns:
            DataFrame con scores ajustados
        """
        preferences = self.get_user_preferences(user_id)
        
        if not preferences or len(recommendations) == 0:
            return recommendations
        
        recommendations = recommendations.copy()
        
        # Ajustar por rango calórico preferido
        if 'preferred_calorie_range' in preferences:
            cal_range = preferences['preferred_calorie_range']
            cal_min, cal_max = cal_range['min'], cal_range['max']
            
            # Bonus para recetas en el rango preferido
            in_range = (recommendations['calories'] >= cal_min) & (recommendations['calories'] <= cal_max)
            recommendations.loc[in_range, 'score'] *= 1.1
        
        # Ajustar por rango de proteína preferido
        if 'preferred_protein_range' in preferences:
            prot_range = preferences['preferred_protein_range']
            prot_min, prot_max = prot_range['min'], prot_range['max']
            
            in_range = (recommendations['protein_per_serving'] >= prot_min) & \
                      (recommendations['protein_per_serving'] <= prot_max)
            recommendations.loc[in_range, 'score'] *= 1.1
        
        # Bonus por categorías preferidas
        if 'preferred_categories' in preferences:
            preferred_cats = preferences['preferred_categories']
            for category, count in preferred_cats.items():
                # Más bonus para categorías más frecuentes
                bonus_multiplier = 1.0 + (count * 0.05)
                mask = recommendations['score_tag'] == category
                recommendations.loc[mask, 'score'] *= bonus_multiplier
        
        # Re-ordenar después de ajustes
        recommendations = recommendations.sort_values('score', ascending=False)
        
        return recommendations
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Dict con estadísticas
        """
        user_file = self.storage_path / f'{user_id}.json'
        
        if not user_file.exists():
            return {
                'total_interactions': 0,
                'recipes_selected': 0,
                'recipes_rejected': 0,
                'average_rating': 0
            }
        
        with open(user_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        interactions = history['interactions']
        
        selected = len([i for i in interactions if i['interaction_type'] == 'selected'])
        rejected = len([i for i in interactions if i['interaction_type'] == 'rejected'])
        ratings = [i['rating'] for i in interactions if i.get('rating')]
        
        return {
            'total_interactions': len(interactions),
            'recipes_selected': selected,
            'recipes_rejected': rejected,
            'average_rating': np.mean(ratings) if ratings else 0,
            'favorite_ingredients': history.get('favorite_ingredients', [])[:5],
            'disliked_ingredients': history.get('disliked_ingredients', [])[:5]
        }
