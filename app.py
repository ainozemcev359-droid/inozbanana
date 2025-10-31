import os, base64, tempfile
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Update, InputFile
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
import httpx

# Загружаем .env локально (на Render берутся из Environment)
load_dotenv()
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not BOT_TOKEN or not GEMINI_KEY:
    # Подскажет, какой именно ключ не найден
    missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN), ("GEMINI_API_KEY", GEMINI_KEY)] if not v]
    raise RuntimeError(f"Missing env: {', '.join(missing)}")

# Эндпоинт Gemini (Nano Banana)
GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
HEADERS = {"Content-Type": "application/json"}

# --- FastAPI + PTB setup ---
app = FastAPI(title="Banana TG Bot")
tg_app = Application.builder().token(BOT_TOKEN).build()

HELP_TEXT = (
    "📸 Пришли фото/скрин и в подписи напиши, что изменить.\n"
    "Примеры:\n"
    "• замени 1.45 на 2.15\n"
    "• поставь Total = 1580\n"
    "• коэффициент 2.35 → 2.95\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — помощь"
)

async def gemini_edit(prompt: str, image_bytes: bytes) -> bytes:
    """
    Свободное редактирование без масок: фото + текст-инструкция.
    Модель старается сохранить стиль/цвет/выравнивание.
    """
    b64_img = base64.b64encode(image_bytes).decode()
    payload = {
        "contents": [{
            "parts": [
                {"text": (
                    "Отредактируй изображение строго по инструкции. "
                    "Сохрани стиль исходного текста: шрифт, кегль, цвет, трекинг, выравнивание. "
                    "Не изменяй остальные области изображения. "
                    "Инструкция: " + prompt
                )},
                {"inline_data": {"mime_type": "image/png", "data": b64_img}}
            ]
        }],
        "generationConfig": {"responseMimeType": "image/png"}
    }
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{GEM_URL}?key={GEMINI_KEY}", headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
        # Достаём base64 с картинкой из ответа
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)

# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я бот на Nano Banana. Пришли мне фото с подписью-инструкцией — и я изменю цифры/текст.\n\n" + HELP_TEXT
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def photo_with_caption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обрабатываем только фото
    if not update.message or not update.message.photo:
        return
    prompt = (update.message.caption or "").strip()
    if not prompt:
        await update.message.reply_text("Добавь подпись к фото — это инструкция, что менять.")
        return

    # Скачиваем фото из Telegram
    tg_file = await update.message.photo[-1].get_file()
    img_bytes = await tg_file.download_as_bytearray()

    # Быстрое ответное сообщение, чтобы пользователь видел прогресс
    msg = await update.message.reply_text("Генерирую…")

    try:
        out_bytes = await gemini_edit(prompt, bytes(img_bytes))
        # Временный файл для отправки обратно
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(out_bytes)
            tmp_path = tmp.name
        await update.message.reply_photo(photo=InputFile(tmp_path))
    except Exception as e:
        # Покажем краткую ошибку пользователю (детали смотри в Render → Logs)
        await msg.edit_text(f"Ошибка: {e}")

def register_handlers(app_):
    app_.add_handler(CommandHandler("start", start))
    app_.add_handler(CommandHandler("help", help_cmd))
    app_.add_handler(MessageHandler(filters.PHOTO, photo_with_caption))

register_handlers(tg_app)

# ---------- Lifecycle: запуск/остановка PTB внутри FastAPI ----------
@app.on_event("startup")
async def _startup():
    # Инициализируем и запускаем Telegram Application
    await tg_app.initialize()
    await tg_app.start()

@app.on_event("shutdown")
async def _shutdown():
    # Аккуратно останавливаем и освобождаем ресурсы
    await tg_app.stop()
    await tg_app.shutdown()

# ---------- Маршруты FastAPI ----------
@app.get("/")
def root():
    # Health-check, чтобы видеть 200 OK на корне
    return {"ok": True, "status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    # Получаем апдейт от Telegram
    data = await request.json()
    # Небольшой лог в Render → Logs (не чувствительные данные)
    try:
        print("WEBHOOK UPDATE:", data.get("update_id"), data.get("message", {}).get("text"))
    except Exception:
        pass
    # Преобразуем в Update и отдаём PTB
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
