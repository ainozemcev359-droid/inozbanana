import os, base64, tempfile, time, json
from io import BytesIO
from typing import Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
import httpx
from PIL import Image

# ---------- ENV ----------
load_dotenv()
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
missing = [n for n, v in [("TG_BOT_TOKEN", BOT_TOKEN), ("GEMINI_API_KEY", GEMINI_KEY)] if not v]
if missing:
    raise RuntimeError(f"Missing env: {', '.join(missing)}")

# ---------- MODELS & HTTP ----------
PRIMARY_MODEL = "gemini-2.5-flash-image"
FALLBACK_MODEL = "gemini-2.0-flash-exp"  # часто доступна в AI Studio
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
HEADERS = {"Content-Type": "application/json"}

# ---------- MIME / IMAGE ----------
def _ensure_image_and_mime(image_bytes: bytes) -> Tuple[bytes, str]:
    """ Возвращает (байты, mime), согласованные с реальным форматом. """
    im = Image.open(BytesIO(image_bytes))
    fmt = (im.format or "").upper()
    if fmt in ("JPEG", "JPG"):
        return image_bytes, "image/jpeg"
    if fmt == "PNG":
        return image_bytes, "image/png"
    # Прочее — конвертируем в PNG, сохраняя альфу если есть
    buf = BytesIO()
    (im if im.mode in ("RGBA", "LA") else im.convert("RGB")).save(buf, format="PNG")
    return buf.getvalue(), "image/png"

def _parts(prompt: str, mime: str, b64data: str):
    """ Базовое содержимое parts без роли (вариант 1/2). """
    return [{
        "parts": [
            {"text": (
                "Отредактируй изображение строго по инструкции. "
                "Сохрани стиль исходного текста (шрифт, размер, цвет, выравнивание, трекинг). "
                "Не изменяй остальные области изображения. "
                "Инструкция: " + prompt
            )},
            {"inline_data": {"mime_type": mime, "data": b64data}}
        ]
    }]

def _parts_with_role(prompt: str, mime: str, b64data: str):
    """ То же, но с явной ролью (вариант 2). """
    return [{
        "role": "user",
        "parts": [
            {"text": (
                "Отредактируй изображение строго по инструкции. "
                "Сохрани стиль исходного текста (шрифт, размер, цвет, выравнивание, трекинг). "
                "Не изменяй остальные области изображения. "
                "Инструкция: " + prompt
            )},
            {"inline_data": {"mime_type": mime, "data": b64data}}
        ]
    }]

async def _post_model(model: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=180) as client:
        url = f"{API_ROOT}/models/{model}:generateContent?key={GEMINI_KEY}"
        r = await client.post(url, headers=HEADERS, json=payload)
        # Для диагностики: если 4xx, вытащим текст
        if r.status_code >= 400:
            try:
                print("GEMINI ERROR BODY:", r.text[:2000])
            except Exception:
                pass
        r.raise_for_status()
        return r.json()

async def gemini_edit(prompt: str, image_bytes: bytes) -> bytes:
    """
    Многоступенчатая попытка вызова Gemini:
    1) 2.5-flash-image + generationConfig.responseMimeType="image/png"
    2) то же + role=user
    3) то же, но БЕЗ generationConfig, c responseModalities=["IMAGE"]
    4) fallback-модель 2.0-flash-exp с (1)
    """
    img1, mime1 = _ensure_image_and_mime(image_bytes)
    b64_1 = base64.b64encode(img1).decode()

    # --- Вариант 1: без роли, с generationConfig ---
    payload1 = {
        "contents": _parts(prompt, mime1, b64_1),
        "generationConfig": {"responseMimeType": "image/png"},
    }
    try:
        data = await _post_model(PRIMARY_MODEL, payload1)
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)
    except httpx.HTTPStatusError as e1:
        if e1.response is None or e1.response.status_code != 400:
            raise

    # --- Вариант 2: добавляем role="user" ---
    payload2 = {
        "contents": _parts_with_role(prompt, mime1, b64_1),
        "generationConfig": {"responseMimeType": "image/png"},
    }
    try:
        data = await _post_model(PRIMARY_MODEL, payload2)
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)
    except httpx.HTTPStatusError as e2:
        if e2.response is None or e2.response.status_code != 400:
            raise

    # --- Вариант 3: без generationConfig, но с responseModalities=["IMAGE"] ---
    payload3 = {
        "contents": _parts_with_role(prompt, mime1, b64_1),
        "responseModalities": ["IMAGE"]
    }
    try:
        data = await _post_model(PRIMARY_MODEL, payload3)
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)
    except httpx.HTTPStatusError as e3:
        if e3.response is None or e3.response.status_code != 400:
            raise

    # --- Вариант 4: fallback-модель ---
    try:
        data = await _post_model(FALLBACK_MODEL, payload1)
        b64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
        return base64.b64decode(b64)
    except httpx.HTTPStatusError as e4:
        # Дадим пользователю короткое тело ошибки, чтобы не было «просто 400»
        body = ""
        try:
            body = e4.response.text[:400]
        except Exception:
            pass
        raise RuntimeError(f"Gemini 400: {body or 'Bad Request'}")

# ---------- FastAPI + PTB ----------
app = FastAPI(title="Banana TG Bot")
tg_app = Application.builder().token(BOT_TOKEN).build()

HELP_TEXT = (
    "📸 Пришли изображение и укажи, что изменить.\n"
    "A) Фото + подпись в одном сообщении (быстро)\n"
    "B) Фото без подписи → потом текст отдельным сообщением (до 10 минут)\n"
    "Можно прислать как «Документ», если это image/*.\n\n"
    "Примеры:\n"
    "• замени 1.45 на 2.15\n"
    "• поставь Total = 1580\n"
    "• коэффициент 2.35 → 2.95\n"
)

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
