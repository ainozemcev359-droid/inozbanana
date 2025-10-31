import os, base64, tempfile, time
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
import httpx
from PIL import Image

# ---------- ENV ----------
load_dotenv()  # локально читает .env; на Render берёт из Environment
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN), ("GEMINI_API_KEY", GEMINI_KEY)] if not v]
if missing:
    raise RuntimeError(f"Missing env: {', '.join(missing)}")

# ---------- Gemini (Nano Banana) ----------
GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
HEADERS = {"Content-Type": "application/json"}

def _ensure_image_and_mime(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Возвращает (байты, mime), согласованные с реальным форматом.
    JPEG/PNG — пропускаем как есть; остальное конвертим в PNG.
    """
    im = Image.open(BytesIO(image_bytes))
    fmt = (im.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        return image_bytes, "image/jpeg"
    if fmt == "PNG":
        return image_bytes, "image/png"
    # Прочие форматы → надёжно в PNG (сохраняем альфу, если есть)
    buf = BytesIO()
    (im if im.mode in ("RGBA", "LA") else im.convert("RGB")).save(buf, format="PNG")
    return buf.getvalue(), "image/png"

async def gemini_edit(prompt: str, image_bytes: bytes) -> bytes:
    """
    Свободное редактирование: фото + текст-инструкция.
    1) детектим mime, 2) при редкой 400 — fallback в PNG.
    """
    img1, mime1 = _ensure_image_and_mime(image_bytes)
    b64_1 = base64.b64encode(img1).decode()

    payload = {
        "contents": [{
            "parts": [
                {"text": (
                    "Отредактируй изображение строго по инструкции. "
                    "Сохрани стиль исходного текста (шрифт, размер, цвет, выравнивание, трекинг). "
                    "Не изменяй остальные области изображения. "
                    "Инструкция: " + prompt
                )},
                {"inline_data": {"mime_type": mime1, "data": b64_1}}
            ]
        }],
        "generationConfig": {"responseMimeType": "image/png"}
    }

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            r = await client.post(f"{GEM_URL}?key={GEMINI_KEY}", headers=HEADERS, json=payload)
            r.raise_for_status()
            data = r.json()
            b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
            return base64.b64decode(b64)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 400:
                # Жёсткий fallback: конверт в PNG и повтор
                im = Image.open(BytesIO(image_bytes))
                buf = BytesIO()
                (im if im.mode in ("RGB", "RGBA") else im.convert("RGBA")).save(buf, format="PNG")
                b64_2 = base64.b64encode(buf.getvalue()).decode()
                payload["contents"][0]["parts"][1]["inline_data"]["mime_type"] = "image/png"
                payload["contents"][0]["parts"][1]["inline_data"]["data"] = b64_2
                r2 = await client.post(f"{GEM_URL}?key={GEMINI_KEY}", headers=HEADERS, json=payload)
                r2.raise_for_status()
                data2 = r2.json()
                b64 = data2["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
                return base64.b64decode(b64)
            raise

# ---------- FastAPI + PTB ----------
app = FastAPI(title="Banana TG Bot")
tg_app = Application.builder().token(BOT_TOKEN).build()

HELP_TEXT = (
    "📸 Пришли изображение и укажи, что изменить.\n"
    "Варианты:\n"
    "A) Фото + подпись в одном сообщении (быстрее)\n"
    "B) Сначала фото без подписи → затем отдельным сообщением текст-инструкция (до 10 минут)\n"
    "Можно прислать как «Документ», если это image/*.\n\n"
    "Примеры:\n"
    "• замени 1.45 на 2.15\n"
    "• поставь Total = 1580\n"
    "• коэффициент 2.35 → 2.95\n"
)

# Память последних фото (user_id -> (bytes, ts))
LAST_PHOTO: dict[int, tuple[bytes, int]] = {}
TTL_SECONDS = 10 * 60

def _set_last_photo(user_id: int, img: bytes) -> None:
    LAST_PHOTO[user_id] = (img, int(time.time()))

def _pop_last_photo(user_id: int) -> Optional[bytes]:
    rec = LAST_PHOTO.get(user_id)
    if not rec: return None
    img, ts = rec
    if int(time.time()) - ts > TTL_SECONDS:
        del LAST_PHOTO[user_id]
        return None
    del LAST_PHOTO[user_id]
    return img

def _has_fresh_photo(user_id: int) -> bool:
    rec = LAST_PHOTO.get(user_id)
    return bool(rec and int(time.time()) - rec[1] <= TTL_SECONDS)

async def _download_best_photo(update: Update) -> Optional[bytes]:
    msg = update.message
    if not msg: return None
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
        return await tg_file.download_as_bytearray()
    if msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
        return await tg_file.download_as_bytearray()
    return None

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Я бот на Nano Banana. Редактирую текст/цифры на картинке.\n\n" + HELP_TEXT)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def on_photo_or_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    caption = (update.message.caption or "").strip() if update.message else ""
    img_bytes = await _download_best_photo(update)
    if not img_bytes:
        return
    if caption:
        msg = await update.message.reply_text("Генерирую…")
        try:
            out = await gemini_edit(caption, bytes(img_bytes))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(out); path = tmp.name
            await update.message.reply_photo(photo=InputFile(path))
        except Exception as e:
            await msg.edit_text(f"Ошибка: {e}")
        return
    _set_last_photo(user_id, bytes(img_bytes))
    await update.message.reply_text("Фото получил ✅ Теперь пришли текст-инструкцию отдельным сообщением (до 10 минут).")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else 0
    text = (update.message.text or "").strip() if update.message else ""
    if not text:
        return
    if not _has_fresh_photo(user_id):
        await update.message.reply_text("Сначала пришли изображение (фото или документ image/*), потом — текст-инструкцию.")
        return
    img_bytes = _pop_last_photo(user_id)
    if not img_bytes:
        await update.message.reply_text("Срок ожидания истёк. Пришли изображение заново.")
        return
    msg = await update.message.reply_text("Генерирую…")
    try:
        out = await gemini_edit(text, bytes(img_bytes))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(out); path = tmp.name
        await update.message.reply_photo(photo=InputFile(path))
    except Exception as e:
        await msg.edit_text(f"Ошибка: {e}")

def register_handlers(app_):
    app_.add_handler(CommandHandler("start", start))
    app_.add_handler(CommandHandler("help", help_cmd))
    app_.add_handler(MessageHandler(filters.PHOTO, on_photo_or_document))
    app_.add_handler(MessageHandler(filters.Document.IMAGE, on_photo_or_document))
    app_.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

register_handlers(tg_app)

# ---------- Lifecycle ----------
@app.on_event("startup")
async def _startup():
    await tg_app.initialize()
    await tg_app.start()

@app.on_event("shutdown")
async def _shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    try:
        print("WEBHOOK UPDATE:", data.get("update_id"))
    except Exception:
        pass
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}
