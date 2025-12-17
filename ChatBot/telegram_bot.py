import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000/ask"  # tu API local

# 🔹 Mensaje de bienvenida
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 Bienvenido al *Asistente Legal*.\n\n"
        "Puedes hacerme preguntas sobre la Ley Orgánica de Transporte Terrestre, "
        "Tránsito y Seguridad Vial y te daré respuestas basadas en los artículos cargados.\n\n"
        "Por ejemplo, prueba escribiendo:\n"
        "➡️ ¿Qué principios generales establece la ley?\n"
        "➡️ ¿Qué dice la ley sobre licencias extranjeras?\n\n"
        "Estoy listo para ayudarte 🚀"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

# 🔹 Manejo de preguntas
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    payload = {"question": user_question, "top_k": 3}
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        answer = data.get("answer", "No se encontró información relevante.")
    except Exception as e:
        answer = f"⚠️ Error al consultar la API: {e}"

    await update.message.reply_text(answer)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # 🔹 Handler para /start
    app.add_handler(CommandHandler("start", start))

    # 🔹 Handler para mensajes normales
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot legal activo en Telegram.")
    app.run_polling()