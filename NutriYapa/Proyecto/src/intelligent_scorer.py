"""
Motor de scoring inteligente para NutriYapa
Sistema de puntuación multi-factor que considera:
- Objetivos nutricionales del usuario
- Perfil metabólico (BMR, TDEE)
- Condiciones de salud
- Preferencias culturales (recetas ecuatorianas)
- Calidad nutricional
"""
import numpy as np
from typing import Dict, Any


class IntelligentScorer:
    """Motor de scoring avanzado para recomendaciones personalizadas"""
    
    def __init__(self):
        # Pesos base por categoría de recomendación
        self.category_weights = {
            # Pérdida de peso
            'optimal_weightloss': 100,
            'very_low_cal_protein': 95,
            'lowcal_highprot': 90,
            'lowcal_highfiber': 85,
            'satiating_lowcal': 85,
            'lowcal_lowfat': 80,
            'lowcal': 75,
            'moderate_weightloss': 65,
            
            # Ganancia muscular
            'optimal_muscle_gain': 100,
            'very_high_protein': 95,
            'high_protein_bulk': 90,
            'high_protein_energy': 85,
            'high_protein': 80,
            'energy_carb_dense': 75,
            'balanced_muscle': 70,
            
            # Bienestar
            'optimal_wellness': 100,
            'nutrient_dense': 95,
            'balanced_healthy': 90,
            'low_fat_healthy': 85,
            'high_fiber_wellness': 85,
            'balanced_protein': 80,
            'balanced': 75,
            'standard': 60,
            
            # General
            'moderate': 50,
            'high_calorie': 30
        }
    
    def score_recipe(self, recipe: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """
        Calcular score total de una receta para un usuario
        
        Args:
            recipe: Dict con características nutricionales de la receta
            user_profile: Dict con perfil completo del usuario
            
        Returns:
            float: Score total (0-100+)
        """
        score = 0.0
        
        # 1. Score base de categoría (30% del total)
        category = recipe.get('score_tag', 'standard')
        score += self.category_weights.get(category, 50) * 0.3
        
        # 2. Alineación con objetivos nutricionales (30% del total)
        score += self._score_macro_alignment(recipe, user_profile) * 0.3
        
        # 3. Calidad nutricional (20% del total)
        score += self._score_nutritional_quality(recipe) * 0.2
        
        # 4. Preferencias culturales y personales (10% del total)
        score += self._score_cultural_preferences(recipe, user_profile) * 0.1
        
        # 5. Condiciones de salud (10% del total)
        score += self._score_health_conditions(recipe, user_profile) * 0.1
        
        # Bonus y penalizaciones adicionales
        score += self._apply_bonus_penalties(recipe, user_profile)
        
        return max(0, score)
    
    def _score_macro_alignment(self, recipe: Dict[str, Any], 
                                user_profile: Dict[str, Any]) -> float:
        """
        Evaluar qué tan bien se alinean los macronutrientes con los objetivos
        """
        score = 50.0  # Score base
        
        macro_targets = user_profile.get('macro_targets')
        if not macro_targets:
            return score
        
        # Obtener macros de la receta y objetivos
        recipe_protein = recipe.get('protein_per_serving', 0)
        recipe_fat = recipe.get('fat', 0)
        recipe_carbs = recipe.get('carbs', 0)
        recipe_calories = recipe.get('calories', 0)
        
        target_protein_g = macro_targets.get('protein_g', 0)
        target_fat_g = macro_targets.get('fat_g', 0)
        target_carbs_g = macro_targets.get('carbs_g', 0)
        target_calories = user_profile.get('target_calories', 2000)
        
        # Calcular cuánto de las necesidades diarias cubre esta comida
        # Asumiendo que es una comida principal (30-40% del día)
        meal_target_protein = target_protein_g * 0.35
        meal_target_fat = target_fat_g * 0.35
        meal_target_carbs = target_carbs_g * 0.35
        meal_target_calories = target_calories * 0.35
        
        # Evaluar cercanía a objetivos (menor desviación = mejor score)
        protein_deviation = abs(recipe_protein - meal_target_protein) / (meal_target_protein + 1)
        fat_deviation = abs(recipe_fat - meal_target_fat) / (meal_target_fat + 1)
        carbs_deviation = abs(recipe_carbs - meal_target_carbs) / (meal_target_carbs + 1)
        calorie_deviation = abs(recipe_calories - meal_target_calories) / (meal_target_calories + 1)
        
        # Score basado en desviaciones (menos desviación = más score)
        protein_score = max(0, 100 - protein_deviation * 100)
        fat_score = max(0, 100 - fat_deviation * 100)
        carbs_score = max(0, 100 - carbs_deviation * 100)
        calorie_score = max(0, 100 - calorie_deviation * 100)
        
        # Pesos según objetivo
        goal = user_profile.get('goal', 'wellness')
        
        if goal == 'lose_weight':
            # Priorizar calorías bajas y proteína alta
            score = (calorie_score * 0.4 + protein_score * 0.4 + 
                    carbs_score * 0.1 + fat_score * 0.1)
        elif goal == 'gain_muscle':
            # Priorizar proteína y calorías adecuadas
            score = (protein_score * 0.5 + calorie_score * 0.3 + 
                    carbs_score * 0.15 + fat_score * 0.05)
        else:  # wellness
            # Balance general
            score = (protein_score * 0.3 + calorie_score * 0.3 + 
                    carbs_score * 0.2 + fat_score * 0.2)
        
        return score
    
    def _score_nutritional_quality(self, recipe: Dict[str, Any]) -> float:
        """
        Evaluar la calidad nutricional general
        """
        score = 50.0
        
        # Densidad nutricional
        nutrient_density = recipe.get('nutrient_density', 0)
        score += min(30, nutrient_density * 50)
        
        # Health score
        health_score = recipe.get('health_score', 0)
        score += min(20, health_score * 0.3)
        
        # Fibra (importante para todos los objetivos)
        fiber = recipe.get('fiber', 0)
        if fiber >= 8:
            score += 15
        elif fiber >= 5:
            score += 10
        elif fiber >= 3:
            score += 5
        
        # Penalizar azúcar alto
        sugar = recipe.get('sugar', 0)
        if sugar > 20:
            score -= 20
        elif sugar > 10:
            score -= 10
        
        # Bonus por carbohidratos de calidad
        carb_quality = recipe.get('carb_quality', 0)
        if carb_quality > 0.5:
            score += 10
        elif carb_quality > 0.2:
            score += 5
        
        return max(0, min(100, score))
    
    def _score_cultural_preferences(self, recipe: Dict[str, Any], 
                                    user_profile: Dict[str, Any]) -> float:
        """
        Evaluar preferencias culturales (recetas ecuatorianas)
        """
        score = 50.0  # Neutral por defecto
        
        # Preferencia por recetas ecuatorianas
        ecuadorian_preference = user_profile.get('ecuadorian_preference', True)
        is_ecuadorian = recipe.get('is_ecuadorian', False)
        ecuadorian_score = recipe.get('ecuadorian_score', 0)
        
        if ecuadorian_preference:
            if is_ecuadorian:
                # Bonus fuerte por recetas ecuatorianas auténticas
                score += 40
                # Bonus adicional según intensidad ecuatoriana
                score += min(10, ecuadorian_score / 10)
            elif ecuadorian_score > 0:
                # Bonus moderado si tiene ingredientes ecuatorianos
                score += min(20, ecuadorian_score / 5)
        else:
            # Si no prefiere ecuatoriano, penalizar levemente
            if is_ecuadorian:
                score -= 10
        
        return max(0, min(100, score))
    
    def _score_health_conditions(self, recipe: Dict[str, Any], 
                                 user_profile: Dict[str, Any]) -> float:
        """
        Evaluar compatibilidad con condiciones de salud
        """
        score = 100.0  # Asumimos compatible por defecto
        
        health_conditions = user_profile.get('health_conditions', [])
        
        for condition in health_conditions:
            # Diabetes: penalizar carbohidratos altos y azúcares
            if condition == 'diabetes':
                carbs = recipe.get('carbs', 0)
                sugar = recipe.get('sugar', 0)
                if sugar > 15:
                    score -= 40
                elif sugar > 10:
                    score -= 20
                if carbs > 60:
                    score -= 30
                elif carbs > 45:
                    score -= 15
            
            # Hipertensión: penalizar sodio alto
            if condition == 'hypertension':
                sodium = recipe.get('sodium', 0)
                if sodium > 800:
                    score -= 50
                elif sodium > 500:
                    score -= 30
            
            # Colesterol alto: penalizar grasas saturadas
            if condition == 'high_cholesterol':
                saturated_fat = recipe.get('saturated_fat', 0)
                if saturated_fat > 15:
                    score -= 40
                elif saturated_fat > 10:
                    score -= 20
            
            # Enfermedad renal: penalizar proteína y sodio alto
            if condition == 'kidney_disease':
                protein = recipe.get('protein_per_serving', 0)
                sodium = recipe.get('sodium', 0)
                if protein > 30:
                    score -= 50
                elif protein > 25:
                    score -= 30
                if sodium > 600:
                    score -= 40
        
        return max(0, score)
    
    def _apply_bonus_penalties(self, recipe: Dict[str, Any], 
                               user_profile: Dict[str, Any]) -> float:
        """
        Aplicar bonus y penalizaciones adicionales
        """
        adjustment = 0.0
        goal = user_profile.get('goal', 'wellness')
        
        # Bonus por eficiencia de proteína
        protein_efficiency = recipe.get('protein_efficiency', 0)
        if goal in ['lose_weight', 'gain_muscle']:
            if protein_efficiency > 10:
                adjustment += 5
            elif protein_efficiency > 7:
                adjustment += 3
        
        # Bonus por índice de saciedad
        satiety_index = recipe.get('satiety_index', 0)
        if goal == 'lose_weight' and satiety_index > 20:
            adjustment += 5
        
        # Penalización por calorías excesivas en pérdida de peso
        if goal == 'lose_weight':
            calories = recipe.get('calories', 0)
            if calories > 600:
                adjustment -= 10
            elif calories > 500:
                adjustment -= 5
        
        # Bonus por balance de macronutrientes
        macro_balance = recipe.get('macro_balance_score', 0)
        if macro_balance > 0.8:
            adjustment += 5
        elif macro_balance > 0.6:
            adjustment += 3
        
        return adjustment
