"""
NutriYapa - Demo Interactiva
Sistema inteligente de recomendaciones nutricionales con IA avanzada
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Configurar path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import NutriRecommender
from src.decision_tree_model import DecisionTreeHeuristic
from src.user_profile import UserProfile, ActivityLevel, Gender, HealthCondition
from src.preference_learner import PreferenceLearner
from src.feature_engineering import compute_nutrient_features, identify_ecuadorian_recipes

# Configuración de la página
st.set_page_config(
    page_title="NutriYapa - Recomendador Nutricional IA",
    page_icon="🥗",
    layout="wide"
)

# Título principal
st.title("🥗 NutriYapa - Tu Asistente Nutricional Inteligente")
st.markdown("### Sistema de recomendaciones con IA personalizada")
st.markdown("---")

# Inicializar sistema de aprendizaje
@st.cache_resource
def get_preference_learner():
    return PreferenceLearner()

learner = get_preference_learner()

# Cargar y procesar datos (con caché)
@st.cache_data
def load_data():
    try:
        recipes = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "recipes.csv")
        # Limpiar datos
        recipes = recipes.dropna(subset=['name', 'calories'])
        
        # Asegurar que existen columnas numéricas básicas
        numeric_cols = ['calories', 'protein', 'fat', 'carbs']
        optional_cols = ['fiber', 'sugar', 'sodium']
        
        # Convertir columnas numéricas existentes
        for col in numeric_cols + optional_cols:
            if col in recipes.columns:
                recipes[col] = pd.to_numeric(recipes[col], errors='coerce')
            else:
                # Si no existe, crear con valor 0
                recipes[col] = 0
        
        recipes = recipes.dropna(subset=['calories', 'protein'])
        return recipes
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# PRE-PROCESAR datos UNA SOLA VEZ (caché para mejor rendimiento)
@st.cache_data
def load_and_preprocess_data():
    """Cargar y pre-procesar recetas con features calculadas"""
    recipes = load_data()
    
    if not recipes.empty:
        # Pre-calcular features nutricionales (esto tarda, pero se cachea)
        recipes = compute_nutrient_features(recipes)
        recipes = identify_ecuadorian_recipes(recipes)
    
    return recipes

recipes_df = load_and_preprocess_data()

if recipes_df.empty:
    st.error("⚠️ No se pudieron cargar los datos. Asegúrate de ejecutar primero prepare_data.py")
    st.stop()

# Sidebar - Configuración del usuario
st.sidebar.header("👤 Tu Perfil Completo")

# Crear tabs en el sidebar
tab1, tab2, tab3 = st.sidebar.tabs(["📋 Básico", "💪 Físico", "🏥 Salud"])

with tab1:
    user_id = st.text_input("ID de Usuario", "usuario_demo")
    
    # Objetivo principal
    goal = st.selectbox(
        "🎯 ¿Cuál es tu objetivo?",
        ["lose_weight", "gain_muscle", "wellness"],
        format_func=lambda x: {
            "lose_weight": "🔥 Bajar de peso",
            "gain_muscle": "💪 Ganar músculo",
            "wellness": "🌟 Bienestar general"
        }[x]
    )
    
    # Preferencia ecuatoriana
    ecuadorian_pref = st.checkbox("🇪🇨 Priorizar recetas ecuatorianas", value=True)

with tab2:
    age = st.number_input("Edad (años)", min_value=15, max_value=100, value=30)
    
    col1, col2 = st.columns(2)
    with col1:
        weight_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    with col2:
        height_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=170)
    
    gender = st.selectbox(
        "Género",
        ["male", "female", "other"],
        format_func=lambda x: {"male": "Masculino", "female": "Femenino", "other": "Otro"}[x]
    )
    
    activity_level = st.selectbox(
        "Nivel de actividad física",
        ["SEDENTARY", "LIGHT", "MODERATE", "ACTIVE", "VERY_ACTIVE"],
        index=2,
        format_func=lambda x: {
            "SEDENTARY": "Sedentario (poco ejercicio)",
            "LIGHT": "Ligero (1-3 días/semana)",
            "MODERATE": "Moderado (3-5 días/semana)",
            "ACTIVE": "Activo (6-7 días/semana)",
            "VERY_ACTIVE": "Muy activo (ejercicio intenso)"
        }[x]
    )

with tab3:
    st.subheader("⚠️ Alergias")
    allergies = st.multiselect(
        "Selecciona tus alergias:",
        ["maní", "lácteos", "huevo", "soya", "trigo", "mariscos", "pescado", "frutos secos"],
        default=[]
    )
    
    st.subheader("❌ No me gusta")
    dislikes = st.multiselect(
        "Ingredientes que prefieres evitar:",
        ["ajo", "cebolla", "cilantro", "champiñones", "aceitunas", "pepinillos", "mayonesa"],
        default=[]
    )
    
    st.subheader("🏥 Condiciones de Salud")
    health_conditions_str = st.multiselect(
        "Selecciona si aplica:",
        ["diabetes", "hypertension", "high_cholesterol", "celiac", 
         "lactose_intolerance", "ibs", "kidney_disease"],
        default=[],
        format_func=lambda x: {
            "diabetes": "Diabetes",
            "hypertension": "Hipertensión",
            "high_cholesterol": "Colesterol alto",
            "celiac": "Celiaquía",
            "lactose_intolerance": "Intolerancia a lactosa",
            "ibs": "Síndrome intestino irritable",
            "kidney_disease": "Enfermedad renal"
        }[x]
    )

# Número de recomendaciones
st.sidebar.markdown("---")
top_k = st.sidebar.slider("📊 Número de recomendaciones", 5, 20, 10)

# Crear perfil de usuario completo
try:
    user_profile = UserProfile(
        user_id=user_id,
        goal=goal,
        age=age,
        weight_kg=weight_kg,
        height_cm=height_cm,
        gender=Gender[gender.upper()],
        activity_level=ActivityLevel[activity_level],
        allergies=allergies,
        dislikes=dislikes,
        health_conditions=[HealthCondition[hc.upper()] for hc in health_conditions_str],
        ecuadorian_preference=ecuadorian_pref
    )
    
    profile_created = True
except Exception as e:
    st.error(f"Error creando perfil: {e}")
    profile_created = False
    user_profile = None

# Mostrar métricas del perfil
if profile_created and user_profile.bmr:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### � Tu Perfil Nutricional")
    
    st.sidebar.metric("Metabolismo en reposo", f"{user_profile.bmr:.0f} kcal")
    st.sidebar.caption("Energía que gastas sin hacer nada")
    
    st.sidebar.metric("Gasto diario total", f"{user_profile.tdee:.0f} kcal")
    st.sidebar.caption("Energía total incluyendo actividad")
    
    st.sidebar.metric("Tu meta calórica diaria", f"{user_profile.target_calories:.0f} kcal")
    if goal == "lose_weight":
        st.sidebar.caption("🔥 Con déficit para bajar de peso")
    elif goal == "gain_muscle":
        st.sidebar.caption("💪 Con superávit para ganar músculo")
    else:
        st.sidebar.caption("🌟 Para mantener tu peso")
    
    if user_profile.macro_targets:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🍽️ Distribución diaria recomendada:**")
        st.sidebar.markdown(f"🥩 Proteína: **{user_profile.macro_targets['protein_g']:.0f}g**")
        st.sidebar.markdown(f"🥑 Grasa: **{user_profile.macro_targets['fat_g']:.0f}g**")
        st.sidebar.markdown(f"🍞 Carbohidratos: **{user_profile.macro_targets['carbs_g']:.0f}g**")

# Sección principal
st.markdown("### 🎯 Dashboard de Nutrición")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Recetas disponibles", f"{len(recipes_df):,}")

with col2:
    goal_emoji = {"lose_weight": "🔥", "gain_muscle": "💪", "wellness": "🌟"}
    st.metric("🎯 Tu objetivo", goal_emoji[goal])

with col3:
    st.metric("🚫 Filtros activos", len(allergies) + len(dislikes) + len(health_conditions_str))

with col4:
    if profile_created and user_profile.target_calories:
        st.metric("🎯 Cal. objetivo", f"{user_profile.target_calories:.0f}")
    else:
        st.metric("🎯 Cal. objetivo", "N/A")

# Botones de acción
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    generate_recs = st.button("✨ Generar Recomendaciones Personalizadas", 
                             type="primary", use_container_width=True)

with col2:
    generate_plan = st.button("📅 Generar Plan Semanal", 
                             type="secondary", use_container_width=True)

# Inicializar session state para mantener recomendaciones
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = None
if 'show_mode' not in st.session_state:
    st.session_state.show_mode = None  # 'recommendations' o 'meal_plan'

# Generar recomendaciones
if generate_recs:
    st.session_state.show_mode = 'recommendations'  # Cambiar a modo recomendaciones
    st.session_state.meal_plan = None  # Limpiar plan semanal
    
    if not profile_created:
        st.error("⚠️ Error en el perfil de usuario. Verifica los datos ingresados.")
    else:
        with st.spinner("🔍 Analizando recetas perfectas para ti..."):
            try:
                # Inicializar recomendador
                recommender = NutriRecommender()
                
                # Obtener recomendaciones (features ya pre-calculadas)
                recommendations = recommender.recommend_for_user(
                    user_profile, 
                    recipes_df, 
                    top_k=top_k,
                    precomputed=True  # ¡Mucho más rápido!
                )
                
                # Aplicar preferencias aprendidas
                recommendations = learner.apply_learned_preferences(user_id, recommendations)
                
                # Guardar en session state para mantener entre recargas
                st.session_state.recommendations = recommendations
                
                if len(recommendations) == 0:
                    st.warning("⚠️ No se encontraron recetas que cumplan con tus criterios. Intenta reducir los filtros.")
                    st.session_state.recommendations = None
                    
            except Exception as e:
                st.error(f"❌ Error generando recomendaciones: {str(e)}")
                st.exception(e)

# Mostrar recomendaciones existentes (si las hay en session_state)
if st.session_state.show_mode == 'recommendations' and st.session_state.recommendations is not None:
    recommendations = st.session_state.recommendations
    
    if len(recommendations) > 0:
        st.success(f"✅ Mostrando {len(recommendations)} recetas recomendadas")
        
        # Mostrar estadísticas de recomendaciones de forma simple
        st.markdown("### 📊 Lo que encontramos para ti")
        st.markdown("*Un resumen rápido de tus recomendaciones personalizadas*")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_calories = recommendations['calories'].mean()
            st.metric("Calorías por porción", f"{avg_calories:.0f} kcal")
            st.caption("Promedio de energía")
        with col2:
            avg_protein = recommendations['protein_per_serving'].mean()
            st.metric("Proteína por porción", f"{avg_protein:.1f}g")
            st.caption("Para tus músculos")
        with col3:
            ecuadorian_count = recommendations['is_ecuadorian'].sum() if 'is_ecuadorian' in recommendations.columns else 0
            st.metric("Recetas ecuatorianas", f"{ecuadorian_count}")
            st.caption("Sabor de casa 🇪🇨")
        with col4:
            match_pct = (len(recommendations) / top_k) * 100
            st.metric("Compatibilidad", f"{match_pct:.0f}%")
            st.caption("Con tus preferencias")
        
        # Mostrar recomendaciones
        st.markdown("### 🍽️ Tus Recetas Recomendadas")
        
        # Diccionario de traducción de categorías
        category_translation = {
            'breakfast': 'Desayuno',
            'lunch': 'Almuerzo',
            'dinner': 'Cena',
            'snack': 'Merienda',
            'appetizer': 'Entrada',
            'dessert': 'Postre',
            'beverage': 'Bebida',
            'salad': 'Ensalada',
            'soup': 'Sopa',
            'main dish': 'Plato principal',
            'side dish': 'Acompañamiento'
        }
        
        for idx, row in recommendations.iterrows():
            # Indicador de receta ecuatoriana
            ecuadorian_badge = " 🇪🇨" if row.get('is_ecuadorian', False) else ""
            
            # Traducir categoría
            category = str(row['score_tag']).lower()
            category_es = category_translation.get(category, category.capitalize())
            
            # Calcular nivel de compatibilidad
            compatibility = (row['score'] / 100) * 100 if row['score'] <= 1 else row['score']
            
            with st.expander(f"**{row['name']}{ecuadorian_badge}** - {compatibility:.0f}% compatible ⭐"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Tipo de comida:** {category_es}")
                    
                    # Mostrar tipo de receta ecuatoriana
                    if row.get('is_ecuadorian', False):
                        ecuadorian_type = row.get('ecuadorian_type', 'general')
                        region_names = {'costa': 'Costa', 'sierra': 'Sierra', 'amazonia': 'Amazonía', 'general': 'Nacional'}
                        region_display = region_names.get(ecuadorian_type.lower(), ecuadorian_type.capitalize())
                        st.markdown(f"**Región:** {region_display}")
                    
                    # Mostrar precio si existe
                    if pd.notna(row.get('precio_aprox')):
                        precio_emoji = {'bajo': '💰', 'medio': '💰💰', 'alto': '💰💰💰'}
                        precio = str(row['precio_aprox']).lower()
                        st.markdown(f"**Precio:** {precio_emoji.get(precio, '')} {precio.capitalize()}")
                    
                    # Mostrar tiempo de preparación si existe
                    if pd.notna(row.get('tiempo_prep')):
                        st.markdown(f"**Tiempo:** ⏱️ {row['tiempo_prep']} minutos")
                    
                    if pd.notna(row.get('description')):
                        st.markdown(f"**Descripción:** {row['description'][:200]}...")
                    
                    # Formatear ingredientes de manera legible
                    if pd.notna(row.get('ingredients')):
                        with st.expander("🛒 Ver ingredientes"):
                            ingredients_raw = str(row['ingredients'])
                            
                            # Limpiar formato JSON/lista de manera más robusta
                            import re
                            # Remover todos los caracteres especiales de formato
                            ingredients_clean = ingredients_raw
                            # Remover paréntesis de formato c(...) o similares
                            ingredients_clean = re.sub(r'\bc\(', '', ingredients_clean)  # c(rice -> rice
                            ingredients_clean = re.sub(r'\)(?=[a-zA-Z])', ' ', ingredients_clean)  # haeo) -> haeo
                            ingredients_clean = re.sub(r'(?<=[a-zA-Z])\)', '', ingredients_clean)  # haeo) -> haeo
                            # Remover caracteres de formato
                            ingredients_clean = ingredients_clean.replace('[', '').replace(']', '')
                            ingredients_clean = ingredients_clean.replace('{', '').replace('}', '')
                            ingredients_clean = ingredients_clean.replace('"', '').replace("'", '')
                            ingredients_clean = ingredients_clean.replace('(', '').replace(')', '')
                            
                            # Separar por comas y limpiar espacios
                            ingredients_list = [ing.strip() for ing in ingredients_clean.split(',') if ing.strip()]
                            # Filtrar ingredientes muy cortos o inválidos
                            ingredients_list = [ing for ing in ingredients_list if len(ing) > 2 and not ing.isdigit()]
                            
                            st.markdown("**Lista de compras:**")
                            for ing in ingredients_list[:20]:  # Limitar a 20 ingredientes
                                st.markdown(f"• {ing}")
                            
                            if len(ingredients_list) > 20:
                                st.caption(f"... y {len(ingredients_list) - 20} ingredientes más")
                    
                    # Botones de interacción (sin recargar recomendaciones)
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("👍 Me gusta", key=f"like_{idx}_cached"):
                            learner.record_interaction(
                                user_id, row['name'], row.to_dict(),
                                'selected', rating=5
                            )
                            st.success("¡Guardado!")
                    with col_b:
                        if st.button("👎 No me gusta", key=f"dislike_{idx}_cached"):
                            learner.record_interaction(
                                user_id, row['name'], row.to_dict(),
                                'rejected', rating=1
                            )
                            st.info("Anotado")
                    with col_c:
                        if st.button("✅ Preparé esta", key=f"made_{idx}_cached"):
                            learner.record_interaction(
                                user_id, row['name'], row.to_dict(),
                                'completed', rating=4
                            )
                            st.success("¡Genial!")
                
                with col2:
                    st.markdown("#### 🍽️ Información Nutricional")
                    st.markdown("*Por porción*")
                    st.markdown(f"")
                    st.markdown(f"🔥 **{row['calories']:.0f}** calorías")
                    st.markdown(f"🥩 **{row['protein_per_serving']:.1f}g** proteínas")
                    st.markdown(f"🥑 **{row.get('fat', 0):.1f}g** grasas")
                    st.markdown(f"🍞 **{row.get('carbs', 0):.1f}g** carbohidratos")
                    
                    if 'fiber' in row and pd.notna(row['fiber']) and row['fiber'] > 0:
                        st.markdown(f"🌾 **{row['fiber']:.1f}g** fibra")
                    
                    st.markdown("---")
                    
                    # Mostrar indicadores de calidad de forma simple
                    if 'nutrient_density' in row and row['nutrient_density'] > 0:
                        quality = "Alta" if row['nutrient_density'] > 5 else "Media" if row['nutrient_density'] > 2 else "Baja"
                        st.markdown(f"✨ Calidad nutricional: **{quality}**")
                    
                    # Indicador de qué tan bien se ajusta a tu objetivo
                    compatibility = (row['score'] / 100) * 100 if row['score'] <= 1 else row['score']
                    if compatibility >= 80:
                        st.success("✅ Excelente para tu objetivo")
                    elif compatibility >= 60:
                        st.info("👍 Buena opción para ti")
                    else:
                        st.warning("⚠️ Opción alternativa")
        
        # Gráfico de distribución de categorías
        st.markdown("---")
        st.markdown("### 📈 Tipos de Comida en tus Recomendaciones")
        st.caption("Esto te muestra qué tipo de comidas te recomendamos más")
        
        # Traducir categorías para el gráfico
        category_counts = recommendations['score_tag'].value_counts()
        category_counts_translated = {}
        for cat, count in category_counts.items():
            cat_lower = str(cat).lower()
            cat_es = category_translation.get(cat_lower, str(cat).capitalize())
            category_counts_translated[cat_es] = count
        
        st.bar_chart(category_counts_translated)
        
        # Opción de descargar resultados
        csv_cols = ['name', 'calories', 'protein_per_serving', 'score_tag', 'score']
        if 'is_ecuadorian' in recommendations.columns:
            csv_cols.append('is_ecuadorian')
        csv = recommendations[csv_cols].to_csv(index=False)
        st.download_button(
            label="📥 Descargar recomendaciones (CSV)",
            data=csv,
            file_name=f"nutriyapa_recomendaciones_{user_id}.csv",
            mime="text/csv"
        )

# Generar plan semanal
if generate_plan:
    st.session_state.show_mode = 'meal_plan'  # Cambiar a modo plan semanal
    st.session_state.recommendations = None  # Limpiar recomendaciones
    
    if not profile_created:
        st.error("⚠️ Error en el perfil de usuario. Verifica los datos ingresados.")
    else:
        with st.spinner("📅 Generando tu plan semanal personalizado..."):
            try:
                recommender = NutriRecommender()
                
                # Generar plan de 7 días
                meal_plan = recommender.get_meal_plan(user_profile, recipes_df, days=7)
                
                # Guardar en session state
                st.session_state.meal_plan = meal_plan
                
            except Exception as e:
                st.error(f"❌ Error generando plan: {str(e)}")
                st.exception(e)

# Mostrar plan semanal existente (si está en session_state)
if st.session_state.show_mode == 'meal_plan' and st.session_state.meal_plan is not None:
    meal_plan = st.session_state.meal_plan
    
    st.success("✅ ¡Plan semanal generado exitosamente!")
    
    st.markdown("### 📅 Tu Plan de Comidas para la Semana")
    st.caption("*Distribución balanceada: Desayuno 25%, Almuerzo 40%, Merienda 25%, Snack 10%*")
    
    # Mostrar plan por día
    for day_key, day_plan in meal_plan.items():
        day_num = day_key.split('_')[1]
        
        with st.expander(f"📆 Día {day_num}", expanded=(day_num == '1')):
            if not day_plan:
                st.warning("No se pudieron generar suficientes recetas para este día")
                continue
            
            # Calcular totales del día
            total_cal = sum([m['calories'] for m in day_plan.values()])
            total_prot = sum([m['protein'] for m in day_plan.values()])
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Mostrar cada comida
                meal_names = {
                    'breakfast': '🌅 Desayuno',
                    'lunch': '🌞 Almuerzo',
                    'dinner': '🌙 Merienda',
                    'snack': '🍎 Snack'
                }
                
                # Diccionario de traducción de categorías para el plan
                meal_category_translation = {
                    'breakfast': 'Desayuno',
                    'lunch': 'Almuerzo', 
                    'dinner': 'Cena',
                    'snack': 'Merienda',
                    'appetizer': 'Entrada',
                    'dessert': 'Postre'
                }
                
                for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
                    if meal_type in day_plan:
                        meal = day_plan[meal_type]
                        category_display = meal_category_translation.get(meal['category'].lower(), meal['category'])
                        st.markdown(f"**{meal_names[meal_type]}:** {meal['name']}")
                        st.caption(f"   {meal['calories']:.0f} kcal | {meal['protein']:.1f}g proteína")
            
            with col2:
                st.markdown("**Total del día:**")
                st.metric("Calorías", f"{total_cal:.0f}")
                st.metric("Proteína", f"{total_prot:.1f}g")
                
                if user_profile.target_calories:
                    diff = total_cal - user_profile.target_calories
                    diff_pct = (diff / user_profile.target_calories) * 100
                    
                    # Mensaje más amigable
                    if abs(diff_pct) <= 5:
                        st.success("✅ ¡Perfecto!")
                    elif diff > 0:
                        st.info(f"📊 +{diff:.0f} kcal")
                    else:
                        st.info(f"📊 {diff:.0f} kcal")
    
    # Resumen semanal
    st.markdown("---")
    st.markdown("### 📊 Resumen de tu Semana")
    st.caption("Un vistazo general a tu plan de alimentación")
    
    total_cals = []
    total_prots = []
    
    for day_plan in meal_plan.values():
        if day_plan:
            total_cals.append(sum([m['calories'] for m in day_plan.values()]))
            total_prots.append(sum([m['protein'] for m in day_plan.values()]))
    
    if total_cals:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Calorías promedio por día", f"{sum(total_cals)/len(total_cals):.0f}")
            st.caption("Tu consumo diario estimado")
        with col2:
            st.metric("Proteína promedio por día", f"{sum(total_prots)/len(total_prots):.1f}g")
            st.caption("Para mantener tus músculos")
        with col3:
            adherence = (sum(total_cals)/len(total_cals)/user_profile.target_calories*100)
            st.metric("Cumplimiento de objetivo", f"{adherence:.0f}%")
            if 95 <= adherence <= 105:
                st.caption("✅ ¡Excelente!")
            else:
                st.caption("📊 Aceptable")
        
        # Gráfico de calorías por día
        st.markdown("### 📈 Calorías de cada día")
        st.caption("Este gráfico muestra cómo se distribuyen tus calorías en la semana")
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Día {i+1}" for i in range(len(total_cals))],
            y=total_cals,
            name='Calorías'
        ))
        
        if user_profile.target_calories:
            fig.add_trace(go.Scatter(
                x=[f"Día {i+1}" for i in range(len(total_cals))],
                y=[user_profile.target_calories] * len(total_cals),
                name='Objetivo',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            yaxis_title="Calorías",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Mostrar estadísticas del usuario
with st.sidebar:
    st.markdown("---")
    st.markdown("### � Tu Historial")
    st.caption("El sistema aprende de tus preferencias")
    
    user_stats = learner.get_user_statistics(user_id)
    
    total_interactions = user_stats.get('total_interactions', 0)
    
    if total_interactions > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recetas vistas", total_interactions)
            st.metric("Me gustaron", user_stats.get('recipes_selected', 0))
        with col2:
            avg_rating = user_stats.get('average_rating', 0)
            rating_emoji = "😊" if avg_rating >= 4 else "😐" if avg_rating >= 3 else "😕"
            st.metric("Satisfacción", f"{rating_emoji}")
            st.metric("No me gustaron", user_stats.get('recipes_rejected', 0))
        
        if user_stats.get('favorite_ingredients'):
            st.markdown("**Te gusta:**")
            for ing in user_stats['favorite_ingredients'][:3]:
                st.markdown(f"• {ing}")
    else:
        st.info("Aún no has interactuado con recetas. ¡Empieza dando likes a las que te gusten!")

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ ¿Cómo funciona?")
st.sidebar.info(
    """
    **NutriYapa** te ayuda a comer mejor:
    
    ✅ **Calcula** cuántas calorías necesitas según tu edad, peso y actividad
    
    ✅ **Recomienda** recetas que se ajustan a tu objetivo (bajar peso, ganar músculo, o mantenerte saludable)
    
    ✅ **Prioriza** recetas ecuatorianas 🇪🇨 con ingredientes que encuentras en tu tienda
    
    ✅ **Respeta** tus alergias y lo que no te gusta
    
    ✅ **Aprende** de tus gustos para mejorar cada vez
    
    💡 **Tip:** Usa los botones 👍👎 para que la app aprenda lo que te gusta
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Desarrollado con ❤️ usando Python, FastAPI y Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
