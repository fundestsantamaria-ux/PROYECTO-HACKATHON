"""
Sistema de perfiles de usuario avanzado para NutriYapa
Incluye cálculos de BMR, TDEE y requerimientos nutricionales personalizados
"""
import numpy as np
from enum import Enum


class ActivityLevel(Enum):
    """Niveles de actividad física para cálculo de TDEE"""
    SEDENTARY = 1.2  # Poco o ningún ejercicio
    LIGHT = 1.375  # Ejercicio ligero 1-3 días/semana
    MODERATE = 1.55  # Ejercicio moderado 3-5 días/semana
    ACTIVE = 1.725  # Ejercicio intenso 6-7 días/semana
    VERY_ACTIVE = 1.9  # Ejercicio muy intenso o trabajo físico


class Gender(Enum):
    """Género para cálculos de BMR"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class HealthCondition(Enum):
    """Condiciones de salud comunes que afectan recomendaciones"""
    DIABETES = "diabetes"
    HYPERTENSION = "hypertension"
    HIGH_CHOLESTEROL = "high_cholesterol"
    CELIAC = "celiac"
    LACTOSE_INTOLERANCE = "lactose_intolerance"
    IBS = "ibs"  # Síndrome de intestino irritable
    KIDNEY_DISEASE = "kidney_disease"


class UserProfile:
    """Perfil de usuario completo con cálculos nutricionales"""
    
    def __init__(self, user_id, goal, age=None, weight_kg=None, height_cm=None,
                 gender=None, activity_level=ActivityLevel.MODERATE,
                 allergies=None, dislikes=None, health_conditions=None,
                 ecuadorian_preference=True, preferred_meal_times=None):
        """
        Inicializar perfil de usuario
        
        Args:
            user_id: ID único del usuario
            goal: Objetivo principal (lose_weight, gain_muscle, wellness)
            age: Edad en años
            weight_kg: Peso en kilogramos
            height_cm: Altura en centímetros
            gender: Género (Gender enum)
            activity_level: Nivel de actividad (ActivityLevel enum)
            allergies: Lista de alergias
            dislikes: Lista de ingredientes no deseados
            health_conditions: Lista de condiciones de salud
            ecuadorian_preference: Preferencia por recetas ecuatorianas
            preferred_meal_times: Horarios preferidos de comidas
        """
        self.user_id = user_id
        self.goal = goal
        self.age = age
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.gender = gender
        self.activity_level = activity_level
        self.allergies = allergies or []
        self.dislikes = dislikes or []
        self.health_conditions = health_conditions or []
        self.ecuadorian_preference = ecuadorian_preference
        self.preferred_meal_times = preferred_meal_times or {}
        
        # Calcular métricas nutricionales
        self.bmr = self._calculate_bmr()
        self.tdee = self._calculate_tdee()
        self.target_calories = self._calculate_target_calories()
        self.macro_targets = self._calculate_macro_targets()
    
    def _calculate_bmr(self):
        """
        Calcular tasa metabólica basal usando ecuación de Mifflin-St Jeor
        BMR = 10*peso(kg) + 6.25*altura(cm) - 5*edad + s
        donde s = 5 para hombres, -161 para mujeres
        """
        if not all([self.age, self.weight_kg, self.height_cm, self.gender]):
            return None
        
        bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age
        
        if self.gender == Gender.MALE:
            bmr += 5
        elif self.gender == Gender.FEMALE:
            bmr -= 161
        else:
            bmr -= 78  # Promedio aproximado
        
        return bmr
    
    def _calculate_tdee(self):
        """
        Calcular gasto energético diario total
        TDEE = BMR * factor de actividad
        """
        if self.bmr is None:
            return None
        
        return self.bmr * self.activity_level.value
    
    def _calculate_target_calories(self):
        """
        Calcular objetivo calórico basado en meta
        - Pérdida de peso: -500 kcal/día (0.5kg/semana)
        - Ganancia muscular: +300 kcal/día
        - Mantenimiento: TDEE
        """
        if self.tdee is None:
            return None
        
        if self.goal == 'lose_weight':
            return self.tdee - 500
        elif self.goal == 'gain_muscle':
            return self.tdee + 300
        else:  # wellness
            return self.tdee
    
    def _calculate_macro_targets(self):
        """
        Calcular objetivos de macronutrientes basados en meta
        
        Pérdida de peso: 40% proteína, 30% grasa, 30% carbohidratos
        Ganancia muscular: 30% proteína, 25% grasa, 45% carbohidratos
        Bienestar: 25% proteína, 30% grasa, 45% carbohidratos
        """
        if self.target_calories is None:
            return None
        
        if self.goal == 'lose_weight':
            protein_pct, fat_pct, carb_pct = 0.40, 0.30, 0.30
        elif self.goal == 'gain_muscle':
            protein_pct, fat_pct, carb_pct = 0.30, 0.25, 0.45
        else:  # wellness
            protein_pct, fat_pct, carb_pct = 0.25, 0.30, 0.45
        
        # Ajustes por condiciones de salud
        if HealthCondition.DIABETES in self.health_conditions:
            # Reducir carbohidratos, aumentar proteína
            carb_pct *= 0.8
            protein_pct += 0.10
        
        if HealthCondition.KIDNEY_DISEASE in self.health_conditions:
            # Reducir proteína
            protein_pct *= 0.7
            carb_pct += 0.15
        
        # Calcular gramos (proteína: 4 kcal/g, grasa: 9 kcal/g, carbohidratos: 4 kcal/g)
        return {
            'protein_g': (self.target_calories * protein_pct) / 4,
            'fat_g': (self.target_calories * fat_pct) / 9,
            'carbs_g': (self.target_calories * carb_pct) / 4,
            'protein_pct': protein_pct,
            'fat_pct': fat_pct,
            'carbs_pct': carb_pct
        }
    
    def get_meal_calorie_target(self, meal_type='main'):
        """
        Obtener objetivo calórico para tipo de comida
        
        Args:
            meal_type: 'breakfast', 'lunch', 'dinner', 'snack'
        """
        if self.target_calories is None:
            return None
        
        # Distribución típica de calorías en Ecuador
        distributions = {
            'breakfast': 0.25,  # Desayuno 25%
            'lunch': 0.40,      # Almuerzo 40% (comida principal)
            'dinner': 0.25,     # Merienda 25%
            'snack': 0.10       # Snack 10%
        }
        
        return self.target_calories * distributions.get(meal_type, 0.33)
    
    def is_suitable_for_condition(self, recipe_features):
        """
        Verificar si una receta es adecuada para las condiciones de salud
        
        Args:
            recipe_features: Dict con características nutricionales
            
        Returns:
            (bool, str): (es_adecuada, razón)
        """
        # Diabetes: limitar carbohidratos y azúcares
        if HealthCondition.DIABETES in self.health_conditions:
            if recipe_features.get('carbs', 0) > 60:
                return False, "Demasiados carbohidratos para diabetes"
            if recipe_features.get('sugar', 0) > 15:
                return False, "Demasiado azúcar para diabetes"
        
        # Hipertensión: limitar sodio
        if HealthCondition.HYPERTENSION in self.health_conditions:
            if recipe_features.get('sodium', 0) > 500:
                return False, "Demasiado sodio para hipertensión"
        
        # Colesterol alto: limitar grasas saturadas
        if HealthCondition.HIGH_CHOLESTEROL in self.health_conditions:
            if recipe_features.get('saturated_fat', 0) > 10:
                return False, "Demasiadas grasas saturadas para colesterol alto"
        
        # Enfermedad renal: limitar proteína, sodio, potasio
        if HealthCondition.KIDNEY_DISEASE in self.health_conditions:
            if recipe_features.get('protein', 0) > 25:
                return False, "Demasiada proteína para enfermedad renal"
            if recipe_features.get('sodium', 0) > 400:
                return False, "Demasiado sodio para enfermedad renal"
        
        return True, "Adecuada"
    
    def to_dict(self):
        """Convertir perfil a diccionario"""
        return {
            'user_id': self.user_id,
            'goal': self.goal,
            'age': self.age,
            'weight_kg': self.weight_kg,
            'height_cm': self.height_cm,
            'gender': self.gender.value if self.gender else None,
            'activity_level': self.activity_level.name if self.activity_level else None,
            'allergies': self.allergies,
            'dislikes': self.dislikes,
            'health_conditions': [c.value for c in self.health_conditions],
            'ecuadorian_preference': self.ecuadorian_preference,
            'bmr': self.bmr,
            'tdee': self.tdee,
            'target_calories': self.target_calories,
            'macro_targets': self.macro_targets
        }
