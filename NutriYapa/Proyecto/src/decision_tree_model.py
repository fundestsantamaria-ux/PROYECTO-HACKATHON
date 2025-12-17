import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


class DecisionTreeWrapper:
    def __init__(self, model=None):
        self.model = model

    def train(self, X, y, max_depth=5):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def export_rules(self, X):
        return export_text(self.model, feature_names=list(X.columns))


class DecisionTreeHeuristic:
    """
    Sistema de clasificación heurística mejorado para recomendaciones nutricionales
    Toma en cuenta objetivos, perfil del usuario y características nutricionales
    """
    
    def __init__(self):
        pass

    def recommend(self, row):
        """
        Recomendar categoría nutricional basada en objetivo y características
        
        Args:
            row: Dict con perfil de usuario y características de la receta
            
        Returns:
            str: Categoría de recomendación
        """
        goal = row.get('goal', 'wellness')
        calories = row.get('calories', 0)
        protein = row.get('protein_per_serving', 0)
        fat = row.get('fat', 0)
        carbs = row.get('carbs', 0)
        fiber = row.get('fiber', 0)
        sugar = row.get('sugar', 0)
        
        # Métricas avanzadas
        protein_ratio = row.get('protein_ratio', 0)
        nutrient_density = row.get('nutrient_density', 0)
        health_score = row.get('health_score', 0)
        carb_quality = row.get('carb_quality', 0)
        
        # Obtener objetivos calóricos del perfil si están disponibles
        target_calories = row.get('target_calories')
        macro_targets = row.get('macro_targets', {})
        
        if goal == 'gain_muscle':
            return self._classify_muscle_gain(
                calories, protein, fat, carbs, fiber,
                protein_ratio, nutrient_density, macro_targets
            )
        elif goal == 'lose_weight':
            return self._classify_weight_loss(
                calories, protein, fat, carbs, fiber, sugar,
                protein_ratio, nutrient_density, carb_quality, macro_targets
            )
        else:  # wellness
            return self._classify_wellness(
                calories, protein, fat, carbs, fiber, sugar,
                health_score, nutrient_density, carb_quality
            )
    
    def _classify_muscle_gain(self, calories, protein, fat, carbs, fiber,
                              protein_ratio, nutrient_density, macro_targets):
        """
        Clasificación para objetivo de ganancia muscular
        Prioriza: Alto en proteína, calorías suficientes, carbohidratos para energía
        """
        target_protein = macro_targets.get('protein_g', 30) if macro_targets else 30
        
        # Óptimo: Alto proteína + calorías adecuadas
        if protein >= target_protein * 0.8 and calories >= 400 and calories <= 700:
            if carbs >= 40 and fiber >= 5:
                return 'optimal_muscle_gain'
            return 'high_protein_bulk'
        
        # Muy alto en proteína
        if protein >= 30:
            return 'very_high_protein'
        
        # Alto en proteína
        if protein >= 20:
            if calories >= 500:
                return 'high_protein_energy'
            return 'high_protein'
        
        # Denso en energía con carbohidratos
        if calories > 500 and carbs > 50:
            return 'energy_carb_dense'
        
        # Balanceado
        if protein >= 15 and protein_ratio >= 0.10:
            return 'balanced_muscle'
        
        return 'moderate'
    
    def _classify_weight_loss(self, calories, protein, fat, carbs, fiber, sugar,
                              protein_ratio, nutrient_density, carb_quality, macro_targets):
        """
        Clasificación para pérdida de peso
        Prioriza: Bajo en calorías, alto en proteína, alto en fibra, bajo en azúcar
        """
        target_protein = macro_targets.get('protein_g', 25) if macro_targets else 25
        
        # Óptimo: Bajo cal + alto prot + alta fibra + bajo azúcar
        if calories < 300 and protein >= 20 and fiber >= 5 and sugar < 10:
            return 'optimal_weightloss'
        
        # Muy bajo en calorías con proteína adecuada
        if calories < 250 and protein >= 15:
            return 'very_low_cal_protein'
        
        # Bajo en calorías, alto en proteína
        if calories < 350 and protein >= 20:
            return 'lowcal_highprot'
        
        # Bajo en calorías con buena fibra
        if calories < 350 and fiber >= 5:
            return 'lowcal_highfiber'
        
        # Saciante (proteína + fibra)
        if protein >= 15 and fiber >= 5 and calories < 400:
            return 'satiating_lowcal'
        
        # Bajo en calorías
        if calories < 400:
            if fat < 10:
                return 'lowcal_lowfat'
            return 'lowcal'
        
        # Moderado si no cumple otros criterios
        if calories < 500 and protein >= 15:
            return 'moderate_weightloss'
        
        return 'high_calorie'
    
    def _classify_wellness(self, calories, protein, fat, carbs, fiber, sugar,
                           health_score, nutrient_density, carb_quality):
        """
        Clasificación para bienestar general
        Prioriza: Balance nutricional, densidad de nutrientes, calidad de ingredientes
        """
        # Óptimo: Balance perfecto
        if (health_score > 50 and nutrient_density > 0.5 and
            300 <= calories <= 500 and protein >= 15 and fiber >= 5):
            return 'optimal_wellness'
        
        # Alta densidad nutricional
        if nutrient_density > 0.6 and fiber >= 5:
            return 'nutrient_dense'
        
        # Balanceado saludable
        if (protein >= 15 and fiber >= 4 and fat < 20 and
            sugar < 15 and 300 <= calories <= 550):
            return 'balanced_healthy'
        
        # Bajo en grasa, saludable
        if fat < 12 and fiber >= 4 and calories < 450:
            return 'low_fat_healthy'
        
        # Alto en fibra
        if fiber >= 8 and carb_quality > 0:
            return 'high_fiber_wellness'
        
        # Proteína moderada, balanceado
        if protein >= 12 and 300 <= calories <= 500:
            return 'balanced_protein'
        
        # Balanceado general
        if 300 <= calories <= 600 and protein >= 10:
            return 'balanced'
        
        return 'standard'