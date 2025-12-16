import streamlit as st
from llm import llm_reply

st.set_page_config(page_title="DiabeTech Assistant", page_icon="🩺", layout="wide")
st.title("DiabeTech Assistant")
st.caption("Formulario clínico + Chat de apoyo (Ollama).")

# =========================
# Dataset schema (real)
# =========================
BIN_FEATURES = [
    ("HighBP", "HighBP (Hipertensión)"),
    ("HighChol", "HighChol (Colesterol alto)"),
    ("CholCheck", "CholCheck (Chequeo colesterol últimos 5 años)"),
    ("Smoker", "Smoker (≥100 cigarrillos en la vida)"),
    ("Stroke", "Stroke (Derrame cerebral previo)"),
    ("HeartDiseaseorAttack", "HeartDiseaseorAttack (Enfermedad coronaria o infarto)"),
    ("PhysActivity", "PhysActivity (Actividad física últimos 30 días)"),
    ("Fruits", "Fruits (Fruta diaria)"),
    ("Veggies", "Veggies (Vegetales diarios)"),
    ("HvyAlcoholConsump", "HvyAlcoholConsump (Alcohol excesivo)"),
    ("AnyHealthcare", "AnyHealthcare (Tiene cobertura de salud)"),
    ("NoDocbcCost", "NoDocbcCost (No vio médico por costo)"),
    ("DiffWalk", "DiffWalk (Dificultad caminar/escaleras)"),
    ("Sex", "Sex (Sexo biológico: 1 Hombre / 0 Mujer)"),
]

NUM_FEATURES = [
    ("BMI", "BMI (Índice de Masa Corporal)", 10.0, 70.0),
    ("MentHlth", "MentHlth (Días mala salud mental: 0–30)", 0, 30),
    ("PhysHlth", "PhysHlth (Días mala salud física: 0–30)", 0, 30),
]

ORD_FEATURES = [
    ("GenHlth", "GenHlth (1=Excelente ... 5=Mala)", 1, 5),
    ("Age", "Age (Categoría 1–14)", 1, 14),
    ("Education", "Education (1–6)", 1, 6),
    ("Income", "Income (1–8)", 1, 8),
]

FEATURE_ORDER = [k for k, _ in BIN_FEATURES] + [k for k, *_ in NUM_FEATURES] + [k for k, *_ in ORD_FEATURES]

# =========================
# Session state
# =========================
if "form" not in st.session_state:
    st.session_state.form = {k: None for k in FEATURE_ORDER}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hola. Soy DiabeTech Assistant.\n\n"
                "Estoy aquí para ayudarte a:\n"
                "- Entender el significado de las variables del formulario\n"
                "- Saber qué valores ingresar (0/1, rangos)\n"
                "- Explicar cómo interpretar el resultado\n\n"
                "Puedes preguntarme, por ejemplo: “¿Qué significa DiffWalk?”"
            ),
        }
    ]

def predict_risk(features: dict) -> dict:
    """
    Placeholder: aquí conectaremos el .keras o pipeline federado.
    Por ahora retorna un resultado demo.
    """
    risk_score = 0.0
    risk_score += 0.8 if features.get("HighBP") == 1 else 0.0
    risk_score += 0.8 if features.get("HighChol") == 1 else 0.0
    risk_score += 0.6 if features.get("Smoker") == 1 else 0.0
    risk_score += 0.7 if features.get("DiffWalk") == 1 else 0.0
    bmi = features.get("BMI") or 0.0
    if bmi >= 30:
        risk_score += 0.9
    elif bmi >= 25:
        risk_score += 0.5

    # normalizar a 0–1
    prob = max(0.0, min(1.0, risk_score / 3.5))

    if prob < 0.33:
        level = "Bajo"
    elif prob < 0.66:
        level = "Moderado"
    else:
        level = "Elevado"

    return {
        "probability": prob,
        "level": level,
        "disclaimer": "Esto es informativo y no reemplaza la evaluación de un profesional de la salud.",
    }

def validate_form(data: dict) -> list[str]:
    errors = []

    # binarios: deben ser 0 o 1
    for k, _ in BIN_FEATURES:
        v = data.get(k)
        if v not in (0, 1):
            errors.append(f"{k}: debe ser 0 o 1.")

    # numéricos
    for k, _, mn, mx in NUM_FEATURES:
        v = data.get(k)
        if v is None:
            errors.append(f"{k}: es requerido.")
            continue
        try:
            fv = float(v)
            if fv < mn or fv > mx:
                errors.append(f"{k}: fuera de rango ({mn}–{mx}).")
        except Exception:
            errors.append(f"{k}: valor inválido.")

    # ordinales
    for k, _, mn, mx in ORD_FEATURES:
        v = data.get(k)
        if v is None:
            errors.append(f"{k}: es requerido.")
            continue
        try:
            iv = int(v)
            if iv < mn or iv > mx:
                errors.append(f"{k}: fuera de rango ({mn}–{mx}).")
        except Exception:
            errors.append(f"{k}: valor inválido (entero).")

    return errors

# =========================
# Layout
# =========================
tab_form, tab_chat = st.tabs(["Formulario clínico", "Chat de apoyo"])

# -------------------------
# TAB: Formulario
# -------------------------
with tab_form:
    st.subheader("Ingreso de datos del paciente")
    st.write("Completa los campos. Luego presiona **Evaluar**.")

    with st.form("patient_form"):
        st.markdown("### Variables binarias (0/1)")
        cols = st.columns(2)
        for i, (k, label) in enumerate(BIN_FEATURES):
            with cols[i % 2]:
                # Radio con valores explícitos 0/1
                val = st.radio(
                    label,
                    options=[0, 1],
                    index=0 if st.session_state.form.get(k) in (None, 0) else 1,
                    horizontal=True,
                    help="0 = No, 1 = Sí (en Sex: 1 Hombre / 0 Mujer)",
                    key=f"form_{k}",
                )
                st.session_state.form[k] = int(val)

        st.markdown("### Variables numéricas")
        cols = st.columns(2)
        for i, (k, label, mn, mx) in enumerate(NUM_FEATURES):
            with cols[i % 2]:
                default = st.session_state.form.get(k)
                if default is None:
                    default = float(mn)
                val = st.number_input(
                    label,
                    min_value=float(mn),
                    max_value=float(mx),
                    value=float(default),
                    step=1.0,
                    help=f"Rango permitido: {mn}–{mx}",
                    key=f"form_{k}",
                )
                st.session_state.form[k] = float(val)

        st.markdown("### Variables categóricas / ordinales")
        cols = st.columns(2)
        for i, (k, label, mn, mx) in enumerate(ORD_FEATURES):
            with cols[i % 2]:
                default = st.session_state.form.get(k)
                if default is None:
                    default = int(mn)
                val = st.number_input(
                    label,
                    min_value=int(mn),
                    max_value=int(mx),
                    value=int(default),
                    step=1,
                    help=f"Rango permitido: {mn}–{mx}",
                    key=f"form_{k}",
                )
                st.session_state.form[k] = int(val)

        colA, colB = st.columns(2)
        with colA:
            submitted = st.form_submit_button("Evaluar", use_container_width=True)
        with colB:
            reset = st.form_submit_button("Limpiar", use_container_width=True)

    if reset:
        st.session_state.form = {k: None for k in FEATURE_ORDER}
        st.rerun()

    if submitted:
        errors = validate_form(st.session_state.form)
        if errors:
            st.error("Corrige lo siguiente antes de evaluar:")
            for e in errors:
                st.write(f"- {e}")
        else:
            result = predict_risk(st.session_state.form)

            st.success("Evaluación generada.")
            st.metric("Riesgo estimado", f"{result['level']}")
            st.write(f"Probabilidad: **{result['probability']:.2f}**")
            st.caption(result["disclaimer"])

            # Informe narrativo con el LLM (opcional, pero útil para demo)
            with st.expander("Generar informe narrativo (pitch)"):
                if st.button("Generar informe", use_container_width=True):
                    ctx = (
                        "El usuario ingresó un formulario clínico. "
                        "Con base en estos valores, genera un informe profesional:\n\n"
                        f"Inputs: {st.session_state.form}\n\n"
                        f"Salida demo: riesgo={result['level']}, prob={result['probability']:.2f}\n\n"
                        "Estructura requerida:\n"
                        "1) Resumen entendible del perfil\n"
                        "2) Interpretación del riesgo (sin diagnóstico) con disclaimer\n"
                        "3) Recomendaciones generales (no médicas)\n"
                        "4) Estimación económica con supuestos explícitos\n"
                    )
                    st.session_state.chat_messages.append({"role": "user", "content": ctx})
                    reply = llm_reply(st.session_state.chat_messages, mode="free")
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    st.write(reply)


with tab_chat:
    st.subheader("Chat de apoyo (explicaciones y dudas)")
    st.write("Úsalo para preguntar sobre variables, formatos, federated learning y la interpretación del demo.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Pregunta aquí (ej: ¿Qué significa NoDocbcCost?)")

    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        reply = llm_reply(st.session_state.chat_messages, mode="free")
        if not reply:
            reply = "No recibí respuesta del modelo. Intenta nuevamente."
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
