# 🥗 NutriYapa - Asistente Nutricional Inteligente con IA

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

**NutriYapa** es un sistema avanzado de recomendaciones nutricionales con inteligencia artificial que te ayuda a alcanzar tus objetivos de salud con recetas personalizadas, especialmente recetas ecuatorianas 🇪🇨.

## 🌟 ¿Qué es NutriYapa?

NutriYapa es tu asistente nutricional digital que te ayuda a alcanzar tus objetivos de salud y bienestar. Ofrecemos recomendaciones de comidas y productos saludables adaptadas a tus metas y preferencias personales, con énfasis en recetas ecuatorianas accesibles y económicas.

## 🎯 Objetivos Soportados

### 🔥 Pérdida de Peso
- Recetas bajas en calorías, altas en proteína y fibra
- Cálculo automático de déficit calórico saludable
- Énfasis en saciedad y nutrientes de calidad

### 💪 Ganancia Muscular
- Recetas altas en proteína y energía
- Superávit calórico calculado para crecimiento muscular
- Balance óptimo de macronutrientes

### 🌟 Bienestar General
- Recetas nutricionalmente balanceadas
- Mantenimiento del peso saludable
- Enfoque en salud integral

## 🚀 Características Principales

### 🧠 Inteligencia Artificial Avanzada

1. **Sistema de Perfiles Completos**
   - Cálculo automático de BMR (Tasa Metabólica Basal)
   - Cálculo de TDEE (Gasto Energético Diario Total)
   - Objetivos calóricos personalizados según actividad física
   - Distribución óptima de macronutrientes

2. **Motor de Scoring Inteligente Multi-Factor**
   - Evaluación de densidad nutricional
   - Alineación con objetivos específicos
   - Consideración de condiciones de salud
   - Pesos dinámicos según prioridades

3. **Feature Engineering Nutricional**
   - Densidad nutricional
   - Calidad de carbohidratos (ratio fibra/azúcar)
   - Eficiencia de proteína
   - Balance de macronutrientes
   - Índice de saciedad estimado

4. **Clasificador de Recetas Ecuatorianas 🇪🇨**
   - Identificación automática de platos típicos:
     - **Costa**: Encebollado, ceviche, encocado, bolón, tigrillo
     - **Sierra**: Locro, fanesca, fritada, hornado, llapingachos
     - **Amazonía**: Maito, ayampaco
   - Detección de ingredientes tradicionales
   - **162 recetas ecuatorianas** con precios accesibles

5. **Sistema de Aprendizaje de Preferencias**
   - Aprende de tus interacciones
   - Mejora recomendaciones con el tiempo
   - Detecta patrones de preferencias

## 🥗 Dataset de Recetas Ecuatorianas

### 📊 Estadísticas del Dataset
- **162 recetas ecuatorianas** totales
- **95 recetas de bajo costo** (59%) - Accesibles en cualquier tienda
- **59 recetas de costo medio** (36%)
- **8 recetas de costo alto** (5%)

### 🍽️ Distribución por Tipo
- **84 almuerzos/cenas** - Platos completos y nutritivos
- **53 snacks/meriendas** - Bocaditos, bebidas y postres
- **25 desayunos** - Opciones energéticas para empezar el día

### 🏪 Enfoque de Accesibilidad
Todas las recetas usan ingredientes disponibles en:
- ✅ Tiendas de la esquina
- ✅ Supermercados locales
- ✅ Mercados municipales

Ingredientes comunes: arroz, plátano, papa, huevos, yuca, mote, queso fresco, pollo, atún, lentejas, y más.

## 📦 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/NutriYapa.git
cd NutriYapa

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🗄️ Preparar Datos

```bash
# Procesar datasets y generar archivos procesados
python script/prepare_data.py
```

Esto procesará:
- ✅ 162 recetas ecuatorianas
- ✅ 55 productos ecuatorianos
- ✅ Dataset general de recetas (opcional)

## 🖥️ Uso

### Demo Interactiva con Streamlit

```bash
streamlit run demo_app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

**Características de la Demo:**
- 👤 Perfil personalizado completo
- 📊 Cálculo automático de BMR y TDEE
- 🎯 Recomendaciones según tu objetivo
- 🇪🇨 Priorización de recetas ecuatorianas
- 💰 Indicador de precio aproximado
- ⏱️ Tiempo de preparación
- 📈 Visualizaciones nutricionales interactivas
- 👍👎 Sistema de feedback para mejorar recomendaciones

### API REST con FastAPI

```bash
python -m uvicorn src.api:app --reload --port 8000
```

La API estará disponible en `http://localhost:8000`
- Documentación interactiva: `http://localhost:8000/docs`

**Ejemplo de uso:**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "goal": "lose_weight",
    "allergies": ["maní", "lácteos"],
    "dislikes": ["champiñones"]
  }'
```

## 📊 Estructura del Proyecto

```
NutriYapa/
├── data/
│   ├── raw/                           # Datos sin procesar
│   │   ├── recetas_ecuatorianas_expandido.csv
│   │   └── productos_ecuatorianos.csv
│   └── processed/                     # Datos procesados
│       ├── recipes.csv
│       └── products.csv
├── models/
│   └── tree.joblib                    # Modelo entrenado
├── src/
│   ├── api.py                         # API FastAPI
│   ├── recommender.py                 # Sistema de recomendaciones
│   ├── intelligent_scorer.py          # Motor de scoring inteligente
│   ├── user_profile.py                # Gestión de perfiles
│   ├── preference_learner.py          # Aprendizaje de preferencias
│   ├── decision_tree_model.py         # Modelo de decisión
│   ├── feature_engineering.py         # Feature engineering
│   ├── data_loader.py                 # Cargador de datos
│   └── config.py                      # Configuración
├── script/
│   ├── prepare_data.py                # Preparación de datos
│   └── train_model.py                 # Entrenamiento
├── tests/
│   └── test_recommender.py            # Tests unitarios
├── demo_app.py                        # Demo Streamlit
├── requirements.txt                   # Dependencias
└── README.md                          # Este archivo
```

## 🧪 Cómo Funciona

### 1. Perfil de Usuario
El sistema calcula automáticamente:
- **BMR** (Tasa Metabólica Basal): Energía que necesitas en reposo
- **TDEE** (Gasto Energético Diario): Energía total considerando actividad física
- **Objetivo Calórico**: Ajustado según tu meta (déficit, superávit o mantenimiento)
- **Macros Objetivo**: Distribución proteína/carbohidratos/grasas personalizada

### 2. Feature Engineering
Se calculan más de 15 métricas nutricionales:
- Densidad nutricional (nutrientes/calorías)
- Ratio proteína/calorías
- Balance de macronutrientes
- Calidad de carbohidratos (fibra/azúcar)
- Índice de saciedad estimado
- Y más...

### 3. Scoring Inteligente Multi-Factor
Cada receta se evalúa con 5 componentes:
- **30% Categoría**: Alineación con tipo de comida deseada
- **30% Macros**: Qué tan cerca está de tus objetivos
- **20% Calidad Nutricional**: Densidad de nutrientes
- **10% Preferencias Culturales**: Bonus para recetas ecuatorianas
- **10% Condiciones de Salud**: Adaptación a restricciones médicas

### 4. Aprendizaje de Preferencias
El sistema aprende de tus interacciones:
- 👍 Likes aumentan preferencias por ingredientes y categorías
- 👎 Dislikes reducen scoring de recetas similares
- Mejora continua de recomendaciones

## 🇪🇨 Recetas Ecuatorianas

### Ejemplos de Recetas Incluidas

**Desayunos:**
- Bolón de verde
- Tigrillo
- Mote pillo
- Colada de avena

**Almuerzos:**
- Encebollado de pescado
- Locro de papa
- Seco de pollo
- Menestra de lentejas
- Arroz con menestra y carne

**Snacks:**
- Empanadas de viento
- Humitas
- Chifles
- Canguil
- Batidos de frutas

### Ingredientes Típicos
- Plátano verde y maduro
- Mote y choclo
- Yuca
- Papa chola
- Queso fresco
- Maní
- Chicharrón
- Cilantro
- Naranjilla, maracuyá, mora

## 🎓 Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **Streamlit**: Interfaz interactiva
- **FastAPI**: API REST moderna
- **Pandas & NumPy**: Procesamiento de datos
- **Scikit-learn**: Machine learning
- **Plotly**: Visualizaciones interactivas

## ⚡ Optimización de Rendimiento

NutriYapa está optimizado para respuesta rápida:

### Caché Inteligente
- ✅ Features nutricionales pre-calculadas al inicio
- ✅ Caché de Streamlit para datos procesados
- ✅ ~10x más rápido en recomendaciones

### Operaciones Vectorizadas
- ✅ Pandas nativo en lugar de `apply()`
- ✅ Filtros con `str.contains()` y regex
- ✅ Scoring por lotes (batch scoring)

### Plan Semanal Optimizado
- ✅ Features calculadas 1 vez para 7 días
- ✅ Indexación por rangos calóricos
- ✅ ~20x más rápido que versión anterior

**Resultado:** Recomendaciones en 1-3 segundos, plan semanal en 3-5 segundos.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.

## 👥 Autores

- **Tu Nombre** - *Desarrollo inicial*

## 🙏 Agradecimientos

- A la comunidad ecuatoriana por compartir sus recetas tradicionales
- A todos los que contribuyen con feedback y mejoras

---

**NutriYapa** - Comida saludable y accesible para todos los ecuatorianos 🥗🇪🇨
2. **Filtrado**: Se eliminan recetas con alérgenos o ingredientes no deseados
3. **Feature Engineering**: Se calculan métricas nutricionales derivadas
4. **Clasificación**: El modelo asigna categorías según el objetivo del usuario
5. **Scoring**: Sistema de puntuación que considera:
   - Categoría de recomendación
   - Ratio proteína/calorías
   - Ajustes por objetivo específico
   - Distancia y precio (cuando disponible)
6. **Rankings**: Se retornan las mejores opciones ordenadas por score

## 🎨 Características de la Demo

- **Perfil de Usuario**: Configura objetivo, alergias y preferencias
- **Recomendaciones en Tiempo Real**: Genera sugerencias personalizadas
- **Visualizaciones**: Gráficos de distribución de categorías
- **Información Nutricional Detallada**: Calorías, proteínas, grasas, carbohidratos
- **Exportación**: Descarga recomendaciones en CSV

## 🛠️ Tecnologías

- **Python 3.8+**
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Machine Learning
- **FastAPI**: API REST
- **Streamlit**: Interfaz web interactiva
- **Joblib**: Persistencia de modelos

## 📈 Categorías de Recomendación

- `optimal_weightloss`: Óptimo para pérdida de peso
- `high_protein_bulk`: Alto en proteína para volumen
- `balanced_healthy`: Balance saludable
- `high_protein`: Alto contenido proteico
- `lowcal_highprot`: Bajo en calorías, alto en proteína
- `low_fat_healthy`: Bajo en grasas, saludable
- `lowcal`: Bajo en calorías
- `energy_dense`: Denso en energía
- `balanced`: Balanceado
- `moderate`: Moderado

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto.

## 👥 Autores

Desarrollado con ❤️ para ayudar a las personas a alcanzar sus objetivos de salud.
