"""
NutriYapa - Tu Asistente Nutricional Inteligente
Sistema de recomendaciones nutricionales personalizado para Ecuador
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go

# Configurar path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import NutriRecommender
from src.user_profile import UserProfile, ActivityLevel, Gender, HealthCondition
from src.preference_learner import PreferenceLearner
from src.feature_engineering import compute_nutrient_features, identify_ecuadorian_recipes

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="NutriYapa - Tu Asistente Nutricional",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN DE ESTADO Y DATOS
# ============================================================================

# Inicializar session state
if 'step' not in st.session_state:
    st.session_state.step = 'welcome'  # welcome, profile, main
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'weekly_plan' not in st.session_state:
    st.session_state.weekly_plan = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Sistema de aprendizaje
@st.cache_resource
def get_preference_learner():
    return PreferenceLearner()

learner = get_preference_learner()

# Cargar datos (solo una vez, en caché)
@st.cache_data
def load_and_preprocess_data():
    """Cargar recetas ecuatorianas en español y pre-procesar"""
    try:
        recipes = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "recipes.csv")
        recipes = recipes.dropna(subset=['name', 'calories'])
        
        # Asegurar columnas numéricas
        numeric_cols = ['calories', 'protein', 'fat', 'carbs', 'fiber', 'sugar', 'sodium']
        for col in numeric_cols:
            if col in recipes.columns:
                recipes[col] = pd.to_numeric(recipes[col], errors='coerce')
            else:
                recipes[col] = 0
        
        recipes = recipes.dropna(subset=['calories', 'protein'])
        
        # Pre-calcular feature
        recipes = compute_nutrient_features(recipes)
        recipes = identify_ecuadorian_recipes(recipes)
        
        return recipes
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

recipes_df = load_and_preprocess_data()

if recipes_df.empty:
    st.error("⚠️ No se pudieron cargar las recetas. Por favor, ejecuta `python script/prepare_data.py` primero.")
    st.stop()

# ============================================================================
# PANTALLA DE BIENVENIDA
# ============================================================================
if st.session_state.step == 'welcome':
    st.markdown('<div class="main-header">🥗 Bienvenido a NutriYapa</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tu asistente nutricional inteligente</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### ¿Qué es NutriYapa?")
        st.markdown("""
        NutriYapa es una herramienta diseñada para ayudarte a tomar mejores decisiones alimenticias 
        de manera simple y práctica.
        
        **¿Qué puedes hacer con NutriYapa?**
        
        ✅ **Recibe recomendaciones personalizadas** según tu objetivo y estilo de vida
        
        ✅ **Descubre recetas ecuatorianas** con ingredientes que conoces y encuentras fácilmente
        
        ✅ **Planifica tu semana** de alimentación sin complicaciones
        
        ✅ **Aprende sobre nutrición** de forma clara y sin tecnicismos
        
        ✅ **Respeta tus preferencias** - alergias, gustos y necesidades especiales
        """)
        
        st.markdown("---")
        
        if st.button("Comenzar", type="primary", use_container_width=True):
            st.session_state.step = 'profile'
            st.rerun()
        
        st.markdown("---")
        
        st.info("💡 **Nota:** Todo el proceso toma menos de 2 minutos y puedes modificar tu información en cualquier momento.")

# ============================================================================
# PANTALLA DE CREACIÓN DE PERFIL (Flujo Guiado)
# ============================================================================
elif st.session_state.step == 'profile':
    st.markdown('<div class="main-header">Cuéntanos sobre ti</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Necesitamos conocerte para darte las mejores recomendaciones</div>', unsafe_allow_html=True)
    
    with st.form("perfil_usuario"):
        st.markdown("### 📋 Información Básica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nombre = st.text_input("¿Cómo te llamas?", placeholder="Ej: María")
            edad = st.number_input("¿Cuántos años tienes?", min_value=15, max_value=100, value=30)
            genero = st.selectbox(
                "Género",
                ["Masculino", "Femenino", "Otro"],
                help="Esto nos ayuda a calcular mejor tus necesidades calóricas"
            )
        
        with col2:
            peso = st.number_input("Tu peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            altura = st.number_input("Tu altura (cm)", min_value=120, max_value=220, value=170)
        
        st.markdown("### 🎯 Tu Objetivo")
        objetivo = st.radio(
            "¿Qué quieres lograr?",
            ["Bajar de peso", "Ganar músculo", "Mantenerme saludable"],
            help="Basado en esto, ajustaremos las calorías y nutrientes recomendados"
        )
        
        st.markdown("### 💪 Actividad Física")
        actividad = st.select_slider(
            "¿Qué tan activo eres?",
            options=[
                "Sedentario (poco o nada de ejercicio)",
                "Ligero (1-3 días/semana)",
                "Moderado (3-5 días/semana)",
                "Activo (6-7 días/semana)",
                "Muy activo (ejercicio intenso diario)"
            ]
        )
        
        st.markdown("### 🚫 Restricciones Alimentarias")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alergias o intolerancias**")
            alergia_lacteos = st.checkbox("Lácteos")
            alergia_huevo = st.checkbox("Huevo")
            alergia_mani = st.checkbox("Maní")
            alergia_mariscos = st.checkbox("Mariscos")
            alergia_soya = st.checkbox("Soya")
        
        with col2:
            st.markdown("**Preferencias alimentarias**")
            prefiere_vegetariano = st.checkbox("Prefiero comidas vegetarianas")
            prefiere_bajo_sodio = st.checkbox("Quiero reducir el sodio")
            prefiere_bajo_azucar = st.checkbox("Quiero reducir el azúcar")
        
        st.markdown("### 🏥 Condiciones de Salud (Opcional)")
        condiciones = st.multiselect(
            "Selecciona si tienes alguna de estas condiciones:",
            [
                "Diabetes",
                "Hipertensión (presión alta)",
                "Colesterol alto",
                "Problemas renales",
                "Problemas digestivos"
            ]
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Continuar", type="primary", use_container_width=True)
        
        if submitted:
            # Mapear valores a los enums
            gender_map = {"Masculino": Gender.MALE, "Femenino": Gender.FEMALE, "Otro": Gender.OTHER}
            goal_map = {
                "Bajar de peso": "lose_weight",
                "Ganar músculo": "gain_muscle",
                "Mantenerme saludable": "wellness"
            }
            activity_map = {
                "Sedentario (poco o nada de ejercicio)": ActivityLevel.SEDENTARY,
                "Ligero (1-3 días/semana)": ActivityLevel.LIGHT,
                "Moderado (3-5 días/semana)": ActivityLevel.MODERATE,
                "Activo (6-7 días/semana)": ActivityLevel.ACTIVE,
                "Muy activo (ejercicio intenso diario)": ActivityLevel.VERY_ACTIVE
            }
            health_map = {
                "Diabetes": HealthCondition.DIABETES,
                "Hipertensión (presión alta)": HealthCondition.HYPERTENSION,
                "Colesterol alto": HealthCondition.HIGH_CHOLESTEROL,
                "Problemas renales": HealthCondition.KIDNEY_DISEASE,
                "Problemas digestivos": HealthCondition.IBS
            }
            
            # Construir lista de alergias
            alergias = []
            if alergia_lacteos: alergias.append("lácteos")
            if alergia_huevo: alergias.append("huevo")
            if alergia_mani: alergias.append("maní")
            if alergia_mariscos: alergias.append("mariscos")
            if alergia_soya: alergias.append("soya")
            
            # Crear perfil
            try:
                user_profile = UserProfile(
                    user_id=nombre.lower().replace(" ", "_") if nombre else "usuario",
                    goal=goal_map[objetivo],
                    age=edad,
                    weight_kg=peso,
                    height_cm=altura,
                    gender=gender_map[genero],
                    activity_level=activity_map[actividad],
                    allergies=alergias,
                    dislikes=[],
                    health_conditions=[health_map[c] for c in condiciones if c in health_map],
                    ecuadorian_preference=True
                )
                
                st.session_state.user_profile = user_profile
                st.session_state.step = 'main'
                st.success("¡Perfil creado exitosamente!")
                st.rerun()
            except Exception as e:
                st.error(f"Error creando perfil: {e}")

# ============================================================================
# PANTALLA PRINCIPAL (Con navegación por secciones)
# ============================================================================
elif st.session_state.step == 'main':
    user_profile = st.session_state.user_profile
    
    # Sidebar con información del usuario
    with st.sidebar:
        st.markdown("### 👤 Tu Perfil")
        st.markdown(f"**Hola, {user_profile.user_id.replace('_', ' ').title()}**")
        
        st.markdown("---")
        
        # Mostrar métricas básicas
        if user_profile.bmr and user_profile.tdee:
            st.markdown("#### 📊 Tus Números")
            st.metric("Meta diaria de calorías", f"{user_profile.target_calories:.0f} kcal")
            
            if user_profile.goal == "lose_weight":
                st.caption("🔥 Déficit para bajar de peso")
            elif user_profile.goal == "gain_muscle":
                st.caption("💪 Superávit para ganar músculo")
            else:
                st.caption("🌟 Mantener peso actual")
            
            st.markdown("---")
            
            if user_profile.macro_targets:
                st.markdown("#### 🍽️ Distribución Recomendada")
                st.markdown(f"🥩 Proteína: **{user_profile.macro_targets['protein_g']:.0f}g**")
                st.markdown(f"🥑 Grasa: **{user_profile.macro_targets['fat_g']:.0f}g**")
                st.markdown(f"🍞 Carbohidratos: **{user_profile.macro_targets['carbs_g']:.0f}g**")
        
        st.markdown("---")
        
        # Mostrar restricciones
        if user_profile.allergies:
            st.markdown("#### ⚠️ Alergias")
            for allergia in user_profile.allergies:
                st.markdown(f"• {allergia}")
        
        if user_profile.health_conditions:
            st.markdown("#### 🏥 Condiciones")
            condition_names = {
                HealthCondition.DIABETES: "Diabetes",
                HealthCondition.HYPERTENSION: "Hipertensión",
                HealthCondition.HIGH_CHOLESTEROL: "Colesterol alto",
                HealthCondition.KIDNEY_DISEASE: "Problemas renales",
                HealthCondition.IBS: "Problemas digestivos"
            }
            for cond in user_profile.health_conditions:
                st.markdown(f"• {condition_names.get(cond, str(cond))}")
        
        st.markdown("---")
        
        if st.button("Editar Perfil", use_container_width=True):
            st.session_state.step = 'profile'
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### ℹ️ ¿Necesitas ayuda?")
        st.info("""
        **Consejos:**
        
        • Marca recetas como favoritas para verlas más seguido
        
        • Indica las que no te gustan para mejorar tus recomendaciones
        
        • Genera un plan semanal para organizarte mejor
        """)
    
    # Header principal
    st.markdown('<div class="main-header">🥗 NutriYapa</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tu plan nutricional personalizado</div>', unsafe_allow_html=True)
    
    # Navegación por pestañas
    tab1, tab2, tab3 = st.tabs([
        "🏠 Recomendaciones del Día",
        "🔍 Explorar Recetas",
        "📅 Plan Semanal"
    ])
    
    # ========================================================================
    # TAB 1: RECOMENDACIONES DEL DÍA
    # ========================================================================
    with tab1:
        st.markdown("### Recetas Recomendadas para Ti")
        st.markdown("Basado en tu objetivo, preferencias y necesidades nutricionales")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_recomendaciones = st.selectbox("¿Cuántas recetas quieres ver?", [5, 10, 15, 20], index=1)
        
        with col2:
            tipo_comida = st.selectbox(
                "Tipo de comida",
                ["Todas", "Desayuno", "Almuerzo", "Snack/Merienda"]
            )
        
        with col3:
            if st.button("🔄 Generar Recomendaciones", type="primary", use_container_width=True):
                with st.spinner("Buscando las mejores recetas para ti..."):
                    try:
                        recommender = NutriRecommender()
                        recommendations = recommender.recommend_for_user(
                            user_profile,
                            recipes_df,
                            top_k=num_recomendaciones,
                            precomputed=True
                        )
                        
                        # Aplicar filtro de tipo de comida
                        if tipo_comida != "Todas":
                            meal_type_map = {
                                "Desayuno": "breakfast",
                                "Almuerzo": "lunch",
                                "Snack/Merienda": "snack"
                            }
                            if tipo_comida in meal_type_map and 'meal_type' in recommendations.columns:
                                recommendations = recommendations[
                                    recommendations['meal_type'] == meal_type_map[tipo_comida]
                                ]
                        
                        # Aplicar preferencias aprendidas
                        recommendations = learner.apply_learned_preferences(
                            user_profile.user_id,
                            recommendations
                        )
                        
                        st.session_state.recommendations = recommendations
                        
                        if len(recommendations) == 0:
                            st.warning("No se encontraron recetas con estos filtros. Intenta con otros criterios.")
                    except Exception as e:
                        st.error(f"Error generando recomendaciones: {e}")
        
        # Mostrar recomendaciones
        if st.session_state.recommendations is not None and len(st.session_state.recommendations) > 0:
            recommendations = st.session_state.recommendations
            
            st.markdown("---")
            
            # Resumen de recomendaciones
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_cals = recommendations['calories'].mean()
                st.metric("Calorías promedio", f"{avg_cals:.0f} kcal")
            
            with col2:
                avg_protein = recommendations['protein_per_serving'].mean()
                st.metric("Proteína promedio", f"{avg_protein:.1f}g")
            
            with col3:
                total_recipes = len(recommendations)
                st.metric("Recetas encontradas", total_recipes)
            
            with col4:
                ecuadorian_count = recommendations.get('is_ecuadorian', pd.Series([False])).sum()
                st.metric("Recetas ecuatorianas", f"{ecuadorian_count} 🇪🇨")
            
            st.markdown("---")
            
            # Mostrar recetas en tarjetas
            for idx, row in recommendations.iterrows():
                with st.container():
                    # Encabezado de la tarjeta
                    col_name, col_actions = st.columns([4, 1])
                    
                    with col_name:
                        ecuadorian_badge = " 🇪🇨" if row.get('is_ecuadorian', False) else ""
                        compatibility = (row['score'] * 100) if row['score'] <= 1 else row['score']
                        st.markdown(f"### {row['name']}{ecuadorian_badge}")
                        st.caption(f"Compatibilidad: {compatibility:.0f}% • {row.get('score_tag', 'Comida').capitalize()}")
                    
                    with col_actions:
                        # Botones de interacción
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            if st.button("💚", key=f"like_{idx}", help="Me gusta"):
                                learner.record_interaction(
                                    user_profile.user_id,
                                    row['name'],
                                    row.to_dict(),
                                    'selected',
                                    rating=5
                                )
                                if row['name'] not in st.session_state.favorites:
                                    st.session_state.favorites.append(row['name'])
                                st.success("¡Guardada!")
                        
                        with col_b:
                            if st.button("👎", key=f"dislike_{idx}", help="No me gusta"):
                                learner.record_interaction(
                                    user_profile.user_id,
                                    row['name'],
                                    row.to_dict(),
                                    'rejected',
                                    rating=1
                                )
                                st.info("Entendido")
                        
                        with col_c:
                            if st.button("✅", key=f"made_{idx}", help="Ya la preparé"):
                                learner.record_interaction(
                                    user_profile.user_id,
                                    row['name'],
                                    row.to_dict(),
                                    'completed',
                                    rating=4
                                )
                                st.success("¡Genial!")
                    
                    # Contenido de la tarjeta
                    col_info, col_nutrition = st.columns([2, 1])
                    
                    with col_info:
                        # Descripción
                        if pd.notna(row.get('description')):
                            st.markdown(f"**Descripción:** {row['description']}")
                        
                        # Información adicional
                        info_parts = []
                        
                        if pd.notna(row.get('tiempo_prep')):
                            info_parts.append(f"⏱️ {row['tiempo_prep']} min")
                        
                        if pd.notna(row.get('precio_aprox')):
                            precio_emoji = {'bajo': '💰', 'medio': '💰💰', 'alto': '💰💰💰'}
                            precio = str(row['precio_aprox']).lower()
                            info_parts.append(f"{precio_emoji.get(precio, '')} {precio.capitalize()}")
                        
                        if pd.notna(row.get('region')):
                            region_names = {
                                'costa': 'Costa',
                                'sierra': 'Sierra',
                                'amazonia': 'Amazonía',
                                'general': 'Nacional'
                            }
                            region = str(row['region']).lower()
                            info_parts.append(f"📍 {region_names.get(region, region.capitalize())}")
                        
                        if info_parts:
                            st.markdown(" • ".join(info_parts))
                        
                        # Ingredientes
                        if pd.notna(row.get('ingredients')):
                            with st.expander("🛒 Ver ingredientes"):
                                import re
                                ingredients_raw = str(row['ingredients'])
                                ingredients_clean = re.sub(r'[c\(\)\[\]\{\}"\']', '', ingredients_raw)
                                ingredients_list = [ing.strip() for ing in ingredients_clean.split(',') if ing.strip() and len(ing.strip()) > 2]
                                
                                for ing in ingredients_list[:15]:
                                    st.markdown(f"• {ing}")
                                
                                if len(ingredients_list) > 15:
                                    st.caption(f"... y {len(ingredients_list) - 15} ingredientes más")
                    
                    with col_nutrition:
                        st.markdown("**Información Nutricional**")
                        st.markdown(f"🔥 **{row['calories']:.0f}** kcal")
                        st.markdown(f"🥩 **{row['protein_per_serving']:.1f}g** proteína")
                        st.markdown(f"🥑 **{row.get('fat', 0):.1f}g** grasa")
                        st.markdown(f"🍞 **{row.get('carbs', 0):.1f}g** carbohidratos")
                        
                        if pd.notna(row.get('fiber')) and row['fiber'] > 0:
                            st.markdown(f"🌾 **{row['fiber']:.1f}g** fibra")
                        
                        # Indicador de calidad
                        if compatibility >= 80:
                            st.success("✅ Excelente opción")
                        elif compatibility >= 60:
                            st.info("👍 Buena opción")
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        elif st.session_state.recommendations is not None:
            st.warning("No se encontraron recetas. Intenta ajustar los filtros.")
    
    # ========================================================================
    # TAB 2: EXPLORAR RECETAS
    # ========================================================================
    with tab2:
        st.markdown("### Explora Todas las Recetas Disponibles")
        st.markdown("Navega por nuestra colección de recetas ecuatorianas")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipo_busqueda = st.selectbox(
                "Tipo de comida",
                ["Todas", "Desayuno", "Almuerzo", "Snack"],
                key="explore_tipo"
            )
        
        with col2:
            precio_filtro = st.selectbox(
                "Precio",
                ["Todos", "Bajo", "Medio", "Alto"],
                key="explore_precio"
            )
        
        with col3:
            busqueda_texto = st.text_input("Buscar por nombre", placeholder="Ej: arroz, sopa, etc.")
        
        # Aplicar filtros
        filtered_recipes = recipes_df.copy()
        
        if tipo_busqueda != "Todas" and 'meal_type' in filtered_recipes.columns:
            meal_type_map = {
                "Desayuno": "breakfast",
                "Almuerzo": "lunch",
                "Snack": "snack"
            }
            if tipo_busqueda in meal_type_map:
                filtered_recipes = filtered_recipes[
                    filtered_recipes['meal_type'] == meal_type_map[tipo_busqueda]
                ]
        
        if precio_filtro != "Todos" and 'precio_aprox' in filtered_recipes.columns:
            filtered_recipes = filtered_recipes[
                filtered_recipes['precio_aprox'].str.lower() == precio_filtro.lower()
            ]
        
        if busqueda_texto:
            filtered_recipes = filtered_recipes[
                filtered_recipes['name'].str.contains(busqueda_texto, case=False, na=False)
            ]
        
        st.markdown(f"**{len(filtered_recipes)} recetas encontradas**")
        
        st.markdown("---")
        
        # Mostrar recetas en formato compacto
        for idx, row in filtered_recipes.head(20).iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                ecuadorian_badge = " 🇪🇨" if row.get('is_ecuadorian', False) else ""
                st.markdown(f"**{row['name']}{ecuadorian_badge}**")
                if pd.notna(row.get('description')):
                    st.caption(row['description'][:100] + ("..." if len(str(row['description'])) > 100 else ""))
            
            with col2:
                st.markdown(f"**{row['calories']:.0f}** kcal")
                st.caption(f"{row['protein_per_serving']:.1f}g proteína")
            
            with col3:
                if st.button("Ver detalles", key=f"explore_{idx}"):
                    st.info(f"**{row['name']}**\n\n{row.get('description', 'Sin descripción')}")
    
    # ========================================================================
    # TAB 3: PLAN SEMANAL
    # ========================================================================
    with tab3:
        st.markdown("### Tu Plan de Alimentación Semanal")
        st.markdown("Genera un plan completo para toda la semana con un solo clic")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📅 Generar Plan de 7 Días", type="primary", use_container_width=True):
                with st.spinner("Creando tu plan personalizado..."):
                    try:
                        recommender = NutriRecommender()
                        meal_plan = recommender.get_meal_plan(
                            user_profile,
                            recipes_df,
                            days=7
                        )
                        st.session_state.weekly_plan = meal_plan
                        st.success("¡Plan semanal generado!")
                    except Exception as e:
                        st.error(f"Error generando plan: {e}")
        
        # Mostrar plan semanal
        if st.session_state.weekly_plan:
            meal_plan = st.session_state.weekly_plan
            
            st.markdown("---")
            
            # Mostrar cada día
            dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            
            for i, (day_key, day_plan) in enumerate(meal_plan.items()):
                if not day_plan:
                    continue
                
                with st.expander(f"📆 {dias_semana[i] if i < 7 else f'Día {i+1}'}", expanded=(i == 0)):
                    # Calcular totales
                    total_cal = sum([m['calories'] for m in day_plan.values()])
                    total_prot = sum([m['protein'] for m in day_plan.values()])
                    
                    # Mostrar cada comida
                    meal_names = {
                        'breakfast': '🌅 Desayuno',
                        'lunch': '🌞 Almuerzo',
                        'dinner': '🌙 Cena',
                        'snack': '🍎 Snack'
                    }
                    
                    for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
                        if meal_type in day_plan:
                            meal = day_plan[meal_type]
                            st.markdown(f"**{meal_names[meal_type]}:** {meal['name']}")
                            st.caption(f"   {meal['calories']:.0f} kcal • {meal['protein']:.1f}g proteína")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total del día", f"{total_cal:.0f} kcal")
                    with col2:
                        st.metric("Proteína total", f"{total_prot:.1f}g")
                    
                    # Comparar con objetivo
                    if user_profile.target_calories:
                        diff = total_cal - user_profile.target_calories
                        diff_pct = abs(diff / user_profile.target_calories) * 100
                        
                        if diff_pct <= 5:
                            st.success("✅ ¡Perfecto! Justo en tu meta")
                        elif diff > 0:
                            st.info(f"📊 {diff:.0f} kcal por encima de tu meta")
                        else:
                            st.info(f"📊 {abs(diff):.0f} kcal por debajo de tu meta")
            
            # Resumen semanal
            st.markdown("---")
            st.markdown("### 📊 Resumen de tu Semana")
            
            total_days_cals = []
            total_days_prots = []
            
            for day_plan in meal_plan.values():
                if day_plan:
                    total_days_cals.append(sum([m['calories'] for m in day_plan.values()]))
                    total_days_prots.append(sum([m['protein'] for m in day_plan.values()]))
            
            if total_days_cals:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_cals = sum(total_days_cals) / len(total_days_cals)
                    st.metric("Promedio diario", f"{avg_cals:.0f} kcal")
                
                with col2:
                    avg_prots = sum(total_days_prots) / len(total_days_prots)
                    st.metric("Proteína diaria", f"{avg_prots:.1f}g")
                
                with col3:
                    if user_profile.target_calories:
                        adherence = (avg_cals / user_profile.target_calories) * 100
                        st.metric("Cumplimiento", f"{adherence:.0f}%")
                
                # Gráfico de calorías por día
                st.markdown("### 📈 Distribución de Calorías")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=[dias_semana[i] if i < 7 else f'Día {i+1}' for i in range(len(total_days_cals))],
                    y=total_days_cals,
                    name='Calorías',
                    marker_color='#2E7D32'
                ))
                
                if user_profile.target_calories:
                    fig.add_trace(go.Scatter(
                        x=[dias_semana[i] if i < 7 else f'Día {i+1}' for i in range(len(total_days_cals))],
                        y=[user_profile.target_calories] * len(total_days_cals),
                        name='Meta',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                
                fig.update_layout(
                    xaxis_title="Día",
                    yaxis_title="Calorías (kcal)",
                    showlegend=True,
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>NutriYapa 🥗 | Tu asistente nutricional inteligente | Hecho en Ecuador 🇪🇨</p>
    <p style='font-size: 0.9rem;'>Desarrollado con ❤️ usando Python y Streamlit</p>
</div>
""", unsafe_allow_html=True)
