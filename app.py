import os, base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
import httpx
from io import BytesIO
import tempfile

load_dotenv()
BOT_TOKEN = os.getenv("8163487098:AAFFhU3-qVEsuWCKuoYseC9QGaUM92M_sc0")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not BOT_TOKEN or not GEMINI_KEY:
    raise RuntimeError("Заполни TG_BOT_TOKEN и GEMINI_API_KEY в .env")

GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
HEADERS = {"Content-Type": "application/json"}

app = FastAPI(title="Banana TG Bot")
tg_app = Application.builder().token(BOT_TOKEN).build()

async def gemini_edit(prompt: str, image_bytes: bytes) -> bytes:
    b64_img = base64.b64encode(image_bytes).decode()
    payload = {
        "contents": [{
            "parts": [
                {"text": (
                    "Отредактируй изображение строго по инструкции. "
                    "Сохрани шрифт, размер, цвет и расположение текста. "
                    "Не изменяй остальное. Инструкция: " + prompt
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
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)

HELP = (
    "📸 Пришли фото со скрином и в подписи укажи, что изменить.\n"
    "Примеры:\n"
    "• замени 1.45 на 2.15\n"
    "• поставь Total = 1580\n"
    "/start — приветствие\n/help — помощь"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я бот на Nano Banana. Пришли фото с подписью — и я изменю цифры/текст. /help")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def photo_with_caption(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        return
    prompt = (update.message.caption or "").strip()
    if not prompt:
        await update.message.reply_text("Добавь подпись к фото — это инструкция, что менять.")
        return

    tg_file = await update.message.photo[-1].get_file()
    img_bytes = await tg_file.download_as_bytearray()
    msg = await update.message.reply_text("Генерирую…")

    try:
        out_bytes = await gemini_edit(prompt, bytes(img_bytes))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(out_bytes)
            tmp_path = tmp.name
        await update.message.reply_photo(photo=InputFile(tmp_path))
    except Exception as e:
        await msg.edit_text(f"Ошибка: {e}")

def register_handlers(app_):
    app_.add_handler(CommandHandler("start", start))
    app_.add_handler(CommandHandler("help", help_cmd))
    app_.add_handler(MessageHandler(filters.PHOTO, photo_with_caption))

register_handlers(tg_app)

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
