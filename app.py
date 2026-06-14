"""cathyAI - Multi-character AI companion web application.

This module provides the main Chainlit application with multi-character support,
live model switching, and optional emotion detection via external APIs.
"""

import os
import httpx
import aiohttp
import asyncpg
import chainlit as cl
try:
    from chainlit.data.chainlit_data_layer import ChainlitDataLayer
except Exception:
    ChainLitDataLayerImportError = True
    ChainlitDataLayer = None
from chainlit.data.storage_clients.base import BaseStorageClient
import json
from urllib.parse import quote
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
CHAT_API_URL = os.getenv("CHAT_API_URL")
MODELS_API_URL = os.getenv("MODELS_API_URL")
EMOTION_API_URL = os.getenv("EMOTION_API_URL")
CHAR_API_URL = os.getenv("CHAR_API_URL", "").rstrip("/")
IDENTITY_API_URL = os.getenv("IDENTITY_API_URL", "").rstrip("/")
CHAT_API_KEY = os.getenv("CHAT_API_KEY")
MODELS_API_KEY = os.getenv("MODELS_API_KEY")
EMOTION_API_KEY = os.getenv("EMOTION_API_KEY")
CHAR_API_KEY = os.getenv("CHAR_API_KEY")
IDENTITY_API_KEY = os.getenv("IDENTITY_API_KEY")
CHAT_TIMEOUT = int(float(os.getenv("CHAT_TIMEOUT", "120")))
MODELS_TIMEOUT = int(float(os.getenv("MODELS_TIMEOUT", "10")))
EMOTION_TIMEOUT = int(float(os.getenv("EMOTION_TIMEOUT", "10")))
EMOTION_ENABLED = os.getenv("EMOTION_ENABLED", "0") == "1"
CHAR_CACHE_PATH = Path("/tmp/characters_cache.json")
CHAR_CACHE_ETAG_PATH = Path("/tmp/characters_cache.etag")
STATE_DIR = Path(os.getenv("STATE_DIR", "/state"))

# Assistant is a first-class Chainlit mode, not a character from the character API.
ASSISTANT_PROFILE_NAME = os.getenv("ASSISTANT_PROFILE_NAME", "Assistant")
ASSISTANT_SYSTEM_PROMPT = os.getenv(
    "ASSISTANT_SYSTEM_PROMPT",
    (
        "You are CathyAI Assistant, a helpful general-purpose assistant. "
        "You are not a roleplay character. "
        "You help with coding, planning, research, troubleshooting, and general questions. "
        "Do not use character diary memory unless explicitly implemented for assistant mode."
    ),
)
ASSISTANT_CHARACTER_IDS = {
    x.strip().lower()
    for x in os.getenv("ASSISTANT_CHARACTER_IDS", "assistant").split(",")
    if x.strip()
}
ASSISTANT_CHARACTER_NAMES = {
    x.strip().lower()
    for x in os.getenv("ASSISTANT_CHARACTER_NAMES", "assistant,cathyai assistant").split(",")
    if x.strip()
}
AUTH_API_URL = os.getenv("AUTH_API_URL", "http://webbui_auth_api:8001").rstrip("/")
AUTH_TIMEOUT = float(os.getenv("AUTH_TIMEOUT", "5"))
USER_ADMIN_API_KEY = os.getenv("USER_ADMIN_API_KEY", "")

# HTTP client
client = httpx.AsyncClient()

def char_headers():
    """Generate headers for character API requests.
    
    :return: Dictionary with API key header if configured
    :rtype: dict
    """
    return {"x-api-key": CHAR_API_KEY} if CHAR_API_KEY else {}

async def fetch_characters_list():
    """Fetch character list from API with ETag caching.
    
    :return: List of character dictionaries from API
    :rtype: list[dict]
    :raises Exception: If API is not configured or request fails
    """
    if not CHAR_API_URL:
        raise Exception("CHAR_API_URL not configured")

    url = f"{CHAR_API_URL}/characters"
    headers = char_headers()
    etag = load_cached_etag()
    if etag:
        headers["If-None-Match"] = etag

    resp = await client.get(url, headers=headers, timeout=10)

    if resp.status_code == 304:
        logger.info("Characters list unchanged (304); using cache")
        return load_cached_characters()

    resp.raise_for_status()

    new_etag = resp.headers.get("etag", "")
    save_cached_etag(new_etag)

    data = resp.json()
    chars = data.get("characters", [])
    CHAR_CACHE_PATH.write_text(json.dumps(chars), encoding="utf-8")
    logger.info(f"Fetched {len(chars)} characters from API (etag={new_etag})")
    return chars

def load_cached_characters():
    """Load characters from local cache file.
    
    :return: List of cached character dictionaries
    :rtype: list[dict]
    """
    if CHAR_CACHE_PATH.exists():
        return json.loads(CHAR_CACHE_PATH.read_text(encoding="utf-8"))
    return []

def load_cached_etag():
    """Load cached ETag from file.
    
    :return: Cached ETag string or empty string
    :rtype: str
    """
    return CHAR_CACHE_ETAG_PATH.read_text(encoding="utf-8").strip() if CHAR_CACHE_ETAG_PATH.exists() else ""

def save_cached_etag(etag: str):
    """Save ETag to cache file.
    
    :param etag: ETag value to cache
    :type etag: str
    """
    if etag:
        CHAR_CACHE_ETAG_PATH.write_text(etag.strip(), encoding="utf-8")

async def fetch_character_private(char_id: str):
    """Fetch full character data with prompts from API with ETag caching.
    
    :param char_id: Character identifier
    :type char_id: str
    :return: Character data with resolved prompts
    :rtype: dict
    :raises Exception: If API request fails
    """
    global CHAR_PRIVATE_ETAGS, CHAR_PRIVATE_CACHE
    url = f"{CHAR_API_URL}/characters/{char_id}?view=private"
    headers = char_headers()
    etag = CHAR_PRIVATE_ETAGS.get(char_id)
    if etag:
        headers["If-None-Match"] = etag

    resp = await client.get(url, headers=headers, timeout=10)
    if resp.status_code == 304:
        cached = CHAR_PRIVATE_CACHE.get(char_id)
        if cached:
            logger.info(f"Character {char_id} not modified (ETag cache hit); using cached private data")
            return cached
        # cache miss: fall back to a normal fetch
        resp = await client.get(url, headers=char_headers(), timeout=10)

    resp.raise_for_status()
    CHAR_PRIVATE_ETAGS[char_id] = resp.headers.get("etag") or ""
    data = resp.json()
    CHAR_PRIVATE_CACHE[char_id] = data
    return data

async def fetch_models():
    """Fetch available models from external API.
    
    :return: List of model names available from the API
    :rtype: list[str]
    """
    if not MODELS_API_URL:
        logger.error("MODELS_API_URL not configured")
        return []
    
    try:
        headers = {"Authorization": f"Bearer {MODELS_API_KEY}"} if MODELS_API_KEY else {}
        response = await client.get(MODELS_API_URL, headers=headers, timeout=MODELS_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        logger.info(f"Fetched {len(models)} models from API")
        return models
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []

async def stream_chat(model, messages):
    """Stream chat responses from external API (Ollama-compatible).
    
    :param model: Name of the model to use for chat
    :type model: str
    :param messages: List of message dictionaries with role and content
    :type messages: list[dict]
    :yield: Token strings from the streaming response
    :rtype: str
    :raises Exception: If API request fails or times out
    """
    if not CHAT_API_URL:
        raise Exception("CHAT_API_URL not configured")
    
    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"
    
    payload = {"model": model, "messages": messages, "stream": True}
    
    try:
        async with client.stream("POST", CHAT_API_URL, json=payload, headers=headers, timeout=CHAT_TIMEOUT) as response:
            response.raise_for_status()
            last = ""
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Ollama NDJSON: {"message":{"content":"..."}, "done":false}
                msg = (chunk.get("message") or {})
                content = msg.get("content")
                if content is not None:
                    # emit only new part
                    if content.startswith(last):
                        delta = content[len(last):]
                    else:
                        delta = content
                    last = content
                    if delta:
                        yield delta
                    continue
                
                # fallback if some other format appears
                if "token" in chunk:
                    yield chunk["token"]
    except httpx.TimeoutException:
        logger.error("Chat API timeout")
        raise Exception("Request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Chat API error: {e}")
        # Fallback to non-streaming
        try:
            payload["stream"] = False
            response = await client.post(CHAT_API_URL, json=payload, headers=headers, timeout=CHAT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if "reply" in data:
                yield data["reply"]
        except Exception as fallback_error:
            logger.error(f"Non-streaming fallback failed: {fallback_error}")
            raise

def extract_ollama_stats(chunk: dict) -> dict:
    """Extract Ollama generation statistics from the final stream chunk."""
    eval_count = chunk.get("eval_count")
    eval_duration = chunk.get("eval_duration")
    prompt_eval_count = chunk.get("prompt_eval_count")
    prompt_eval_duration = chunk.get("prompt_eval_duration")
    total_duration = chunk.get("total_duration")
    load_duration = chunk.get("load_duration")

    tok_s = None
    if eval_count and eval_duration:
        tok_s = eval_count / (eval_duration / 1_000_000_000)

    return {
        "prompt_tokens": prompt_eval_count,
        "completion_tokens": eval_count,
        "tok_s": tok_s,
        "total_time_s": total_duration / 1_000_000_000 if total_duration else None,
        "load_time_s": load_duration / 1_000_000_000 if load_duration else None,
        "prompt_eval_time_s": prompt_eval_duration / 1_000_000_000 if prompt_eval_duration else None,
        "eval_time_s": eval_duration / 1_000_000_000 if eval_duration else None,
    }


async def stream_chat_events_callback(model, messages, on_event):
    """Stream chat using aiohttp to avoid httpx GeneratorExit cleanup noise."""
    if not CHAT_API_URL:
        raise Exception("CHAT_API_URL not configured")

    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"

    payload = {"model": model, "messages": messages, "stream": True}

    timeout = aiohttp.ClientTimeout(total=CHAT_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(CHAT_API_URL, json=payload, headers=headers) as response:
                if response.status >= 400:
                    text = await response.text()
                    logger.error("Chat API error %s: %s", response.status, text[:500])
                    raise Exception(f"Chat API error {response.status}")

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        line = line[6:]

                    if line == "[DONE]":
                        return

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON stream line: %s", line[:200])
                        continue

                    msg = chunk.get("message") or {}

                    thinking = msg.get("thinking")
                    if thinking:
                        await on_event({
                            "type": "thinking_delta",
                            "content": thinking,
                            "raw": chunk,
                        })

                    content = msg.get("content")
                    if content:
                        await on_event({
                            "type": "answer_delta",
                            "content": content,
                            "raw": chunk,
                        })

                    if chunk.get("done"):
                        await on_event({
                            "type": "stats",
                            "data": extract_ollama_stats(chunk),
                            "raw": chunk,
                        })
                        return

                    if "token" in chunk:
                        await on_event({
                            "type": "answer_delta",
                            "content": chunk["token"],
                            "raw": chunk,
                        })

    except TimeoutError:
        logger.error("Chat API timeout")
        raise Exception("Request timed out")
    except aiohttp.ClientError as e:
        logger.error("Chat API aiohttp error: %s", e)
        raise


async def stream_chat_events(model, messages):
    """Compatibility wrapper for existing async-for consumers."""
    queue = asyncio.Queue()
    sentinel = object()

    async def on_event(event):
        await queue.put(event)

    async def runner():
        try:
            await stream_chat_events_callback(model, messages, on_event)
        except Exception as e:
            await queue.put({"type": "error", "error": e})
        finally:
            await queue.put(sentinel)

    task = asyncio.create_task(runner())

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, dict) and item.get("type") == "error":
                raise item["error"]
            yield item
    finally:
        if not task.done():
            task.cancel()



async def detect_emotion(text):
    """Detect emotion from text using external API.
    
    :param text: Text content to analyze for emotion
    :type text: str
    :return: Dictionary with emotion label and confidence score, or None if disabled/failed
    :rtype: dict or None
    """
    if not EMOTION_ENABLED or not EMOTION_API_URL:
        return None
    
    try:
        headers = {"Content-Type": "application/json"}
        if EMOTION_API_KEY:
            headers["Authorization"] = f"Bearer {EMOTION_API_KEY}"
        
        response = await client.post(EMOTION_API_URL, json={"text": text}, headers=headers, timeout=EMOTION_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {"label": data.get("label"), "score": data.get("score")}
    except Exception as e:
        logger.warning(f"Emotion detection failed: {e}")
        return None

async def identity_resolve(external_id: str):
    """Resolve external user ID to identity data.
    
    :param external_id: External identifier (e.g. chainlit:username:alice)
    :type external_id: str
    :return: Identity data with person_id and preferred_name, or empty dict if unavailable
    :rtype: dict
    """
    if not IDENTITY_API_URL:
        return {}
    headers = {"x-api-key": IDENTITY_API_KEY} if IDENTITY_API_KEY else {}
    try:
        r = await client.get(
            f"{IDENTITY_API_URL}/identity/resolve",
            params={"external_id": external_id},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Identity resolve failed for {external_id}: {e}")
        return {}

async def identity_link(external_id: str, preferred_name: str):
    """Link external user ID to identity with preferred name.
    
    :param external_id: External identifier to link
    :type external_id: str
    :param preferred_name: Preferred display name for user
    :type preferred_name: str
    :return: Identity data from link operation, or empty dict if failed
    :rtype: dict
    """
    if not IDENTITY_API_URL:
        return {}
    headers = {"x-api-key": IDENTITY_API_KEY} if IDENTITY_API_KEY else {}
    try:
        r = await client.post(
            f"{IDENTITY_API_URL}/identity/link",
            json={"external_id": external_id, "preferred_name": preferred_name},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        return r.json() if r.text else {}
    except Exception as e:
        logger.warning(f"Identity link failed for {external_id}: {e}")
        return {}

async def identity_ensure(external_id: str, username: str | None):
    """Ensure identity exists, creating if necessary.
    
    Attempts to resolve identity, and if not found (404), creates a link
    and resolves again. Provides auto-provisioning for new users.
    
    :param external_id: External identifier to ensure
    :type external_id: str
    :param username: Username to use as fallback preferred name
    :type username: str or None
    :return: Identity data with person_id and preferred_name, or empty dict if failed
    :rtype: dict
    """
    if not IDENTITY_API_URL:
        return {}
    headers = {"x-api-key": IDENTITY_API_KEY} if IDENTITY_API_KEY else {}
    try:
        r = await client.get(
            f"{IDENTITY_API_URL}/identity/resolve",
            params={"external_id": external_id},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 404:
            # Create/link then resolve again
            await identity_link(external_id, username or "there")
            r = await client.get(
                f"{IDENTITY_API_URL}/identity/resolve",
                params={"external_id": external_id},
                headers=headers,
                timeout=10,
            )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Identity ensure failed for {external_id}: {e}")
        return {}

# Load characters from API or cache
CHAR_INDEX = {}
CHAR_LIST = []
CHAR_PRIVATE_ETAGS = {}
CHAR_PRIVATE_CACHE = {}
PROFILE_NAME_TO_ID = {}

# Validate configuration
if not CHAR_API_URL:
    logger.warning("CHAR_API_URL not configured")
if not CHAR_API_KEY:
    logger.warning("CHAR_API_KEY not configured (character-api may reject requests)")

def session_id() -> str:
    """Get current session ID.
    
    :return: Session identifier with chainlit prefix
    :rtype: str
    """
    sid = cl.user_session.get("id") or "unknown"
    return f"chainlit:{sid}"

def character_display_name(char: dict) -> str:
    """Get character display name: nickname > first name > Assistant.
    
    :param char: Character dictionary
    :type char: dict
    :return: Display name for character
    :rtype: str
    """
    if not isinstance(char, dict):
        return "Assistant"
    nickname = char.get("nickname")
    if isinstance(nickname, str) and nickname.strip():
        return nickname.strip()
    name = char.get("name")
    if isinstance(name, str) and name.strip():
        first = name.strip().split(" ")[0].strip()
        if first:
            return first
    return "Assistant"

def character_author_name(char: dict) -> str:
    """Get exact chat profile name for Chainlit avatar matching.
    
    :param char: Character dictionary
    :type char: dict
    :return: Full character name for author field
    :rtype: str
    """
    if not isinstance(char, dict):
        return "Assistant"
    name = char.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return character_display_name(char)

async def send_character_message(content: str, char: dict) -> cl.Message:
    """Send message using built-in Chainlit avatar via author/profile-name match.
    
    :param content: Message content
    :type content: str
    :param char: Character dictionary
    :type char: dict
    :return: Sent message object
    :rtype: cl.Message
    """
    msg = cl.Message(content=content, author=character_display_name(char))
    await msg.send()
    return msg

async def register_character_avatar(char: dict):
    """Register Chainlit avatar for this character's message author name.
    
    :param char: Character dictionary with avatar info
    :type char: dict
    """
    author_name = character_display_name(char)

    avatar_url = (char.get("avatar_url") or "").strip()
    if not avatar_url and CHAR_API_URL:
        avatar = str(char.get("avatar") or "").strip()
        if avatar:
            avatar_url = f"{CHAR_API_URL}/avatars/{avatar}"

    if avatar_url:
        try:
            avatar = cl.Avatar(name=author_name, url=avatar_url)
            await avatar.send()
            logger.info("Registered avatar for author=%r url=%r", author_name, avatar_url)
        except Exception as e:
            logger.warning("Failed to register avatar for %r: %s", author_name, e)

def is_admin() -> bool:
    """Check if current user has admin role.
    
    :return: True if user is admin, False otherwise
    :rtype: bool
    """
    return cl.user_session.get("auth_role") == "admin"

async def require_admin_or_warn() -> bool:
    """Require admin role or send warning message.
    
    :return: True if user is admin, False otherwise
    :rtype: bool
    """
    if is_admin():
        return True
    await cl.Message(content="❌ Admin only.").send()
    return False

def _admin_headers():
    """Generate headers for admin API requests.
    
    :return: Dictionary with x-admin-key header if configured
    :rtype: dict
    """
    return {"x-admin-key": USER_ADMIN_API_KEY} if USER_ADMIN_API_KEY else {}

def append_event(sender: str, text: str):
    """Append conversation event to session log.
    
    Creates NDJSON log files in /state/sessions/<person_id>/<char_id>/<session_id>.ndjson
    for persistent conversation history. Gracefully handles failures without disrupting chat.
    
    :param sender: Message sender (user, assistant, or system)
    :type sender: str
    :param text: Message content
    :type text: str
    """
    try:
        pid = cl.user_session.get("person_id") or "unknown_person"
        cid = cl.user_session.get("char_id") or "unknown_char"
        eid = cl.user_session.get("external_user_id") or "unknown"
        sid = session_id()
        p = STATE_DIR / "sessions" / pid / cid
        p.mkdir(parents=True, exist_ok=True)
        f = p / f"{sid.replace(':', '_')}.ndjson"
        evt = {
            "ts": int(time.time() * 1000),
            "source": "chainlit",
            "session_id": sid,
            "person_id": pid,
            "char_id": cid,
            "external_user_id": eid,
            "sender": sender,
            "text": text,
            "len": len(text),
        }
        with f.open("a", encoding="utf-8") as w:
            w.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to append event: {e}")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate user via auth API.
    
    :param username: Username to authenticate
    :type username: str
    :param password: Password to verify
    :type password: str
    :return: User object if authenticated, None otherwise
    :rtype: cl.User or None
    """
    logger.info(f"[AUTH] login attempt username={username!r}")
    try:
        with httpx.Client(timeout=AUTH_TIMEOUT) as c:
            r = c.post(
                f"{AUTH_API_URL}/auth/login",
                json={"username": username, "password": password},
            )
        if r.status_code != 200:
            logger.info(f"[AUTH] failed status={r.status_code}")
            return None
        
        data = r.json()
        role = data.get("role", "user")
        logger.info(f"[AUTH] ok role={role}")
        
        return cl.User(identifier=username, metadata={"role": role})
    except Exception as e:
        logger.exception(f"[AUTH] auth_api error: {e}")
        return None


def is_assistant_character_record(char: dict) -> bool:
    """Return True if a character API record is the old Assistant pseudo-character."""
    if not isinstance(char, dict):
        return False

    cid = str(char.get("id") or "").strip().lower()
    name = str(char.get("name") or "").strip().lower()
    kind = str(char.get("kind") or char.get("type") or "").strip().lower()

    return (
        kind == "assistant"
        or cid in ASSISTANT_CHARACTER_IDS
        or name in ASSISTANT_CHARACTER_NAMES
    )


def assistant_system_text(preferred_name: str) -> str:
    """Build the Assistant-mode system prompt."""
    identity_hint = (
        f"You are chatting with a user whose preferred name is '{preferred_name}'. "
        f"Address them as '{preferred_name}' when natural, but do not overuse their name. "
        f"Do not ask them what their name is - you already know it.\n\n"
    )
    return identity_hint + ASSISTANT_SYSTEM_PROMPT



def is_embedding_model(model_name: str) -> bool:
    """Return True if a model is embedding-only and should not be used for chat."""
    name = (model_name or "").lower()
    return (
        "embed" in name
        or "embedding" in name
        or name.startswith("nomic-embed")
        or name.startswith("mxbai-embed")
    )


def choose_default_chat_model(model_names: list[str]) -> str | None:
    """Pick a sensible default chat model, avoiding embedding models."""
    usable = [m for m in model_names if m and not is_embedding_model(m)]
    if not usable:
        return None

    preferred = [
        "qwen3:32b",
        "qwen3:8b",
        "qwen3:latest",
        "qwen2.5-coder:32b",
        "qwen3-coder:30b",
        "devstral:latest",
    ]

    for wanted in preferred:
        if wanted in usable:
            return wanted

    return usable[0]


async def send_model_settings(model_names: list[str], default_model: str | None, thinking: str = "Auto"):
    """Send model selector + thinking toggle settings."""
    values = [m for m in model_names if m]
    if not values:
        return

    initial = default_model or values[0]
    if initial not in values:
        initial = values[0]

    thinking_values = ["Auto", "On", "Off"]
    if thinking not in thinking_values:
        thinking = "Auto"

    try:
        await cl.ChatSettings(
            [
                cl.input_widget.Select(
                    id="Model",
                    label="Ollama Model",
                    values=values,
                    initial_value=initial,
                ),
                cl.input_widget.Select(
                    id="Thinking",
                    label="Thinking",
                    values=thinking_values,
                    initial_value=thinking,
                ),
            ]
        ).send()
    except Exception as e:
        logger.error(f"Failed to send chat settings: {e}")



def get_active_thread_id():
    """Return the selected persisted Chainlit thread id.

    Do not use cl.context.session.id here; that is a socket/session id,
    not the persisted Thread.id in Postgres.
    """
    for key in ("loaded_thread_id", "resumed_thread_id", "current_thread_id"):
        val = cl.user_session.get(key)
        if val:
            return str(val)

    return None


async def ensure_assistant_history_for_active_thread(preferred_name: str):
    """Load DB history for the current active thread when switching/reloading."""
    thread_id = get_active_thread_id()
    loaded_thread_id = cl.user_session.get("loaded_thread_id")

    if not thread_id:
        logger.warning("Assistant active thread id unavailable; using in-memory history")
        return

    if loaded_thread_id == thread_id and cl.user_session.get("history"):
        return

    history = await fetch_assistant_thread_history_from_db_v2(thread_id, preferred_name)
    cl.user_session.set("history", history)
    cl.user_session.set("loaded_thread_id", thread_id)

    logger.info(
        f"Loaded active Assistant thread {thread_id} with {max(len(history)-1, 0)} model history messages"
    )



async def handle_assistant_message(message: cl.Message):
    """Handle Assistant-mode messages without requiring a character."""
    model_available = cl.user_session.get("model_available", False)
    if not model_available:
        await cl.Message(content="⚠ No chat models available. Please check API configuration.").send()
        return

    settings = cl.user_session.get("settings", {})
    default_model = cl.user_session.get("default_model")
    selected_model = settings.get("Model", default_model)

    if not selected_model or is_embedding_model(selected_model):
        await cl.Message(
            content=(
                "⚠ The selected model is not a chat model. "
                "Please select a chat model such as `qwen3:32b`, `qwen3:8b`, or `qwen2.5-coder:32b`."
            )
        ).send()
        return

    preferred_name = cl.user_session.get("preferred_name", "there")
    await ensure_assistant_history_for_active_thread(preferred_name)

    history = cl.user_session.get("history") or []
    if not history:
        history = [{"role": "system", "content": assistant_system_text(preferred_name)}]

    history.append({"role": "user", "content": message.content})
    append_event("user", message.content)

    reply = ""
    stats_data = None
    thinking_text = ""
    last_reasoning_update = 0.0
    reasoning_update_interval = 0.25

    msg = cl.Message(content="", author=ASSISTANT_PROFILE_NAME)
    await msg.send()

    reasoning_element = cl.CustomElement(
        name="ReasoningPanel",
        props={
            "thinking": "",
            "isThinking": True,
            "stats": {},
        },
        display="inline",
    )

    msg.elements = [reasoning_element]
    await msg.update()

    async def on_assistant_event(event):
        nonlocal reply, stats_data, thinking_text, last_reasoning_update

        event_type = event.get("type")

        if event_type == "thinking_delta":
            delta = event.get("content") or ""
            if not delta:
                return

            thinking_text += delta

            now = time.monotonic()
            if now - last_reasoning_update >= reasoning_update_interval:
                reasoning_element.props = {
                    "thinking": thinking_text,
                    "isThinking": True,
                    "stats": stats_data or {},
                }
                msg.elements = [reasoning_element]
                await msg.update()
                last_reasoning_update = now

        elif event_type == "answer_delta":
            delta = event.get("content") or ""
            if not delta:
                return

            if thinking_text and reasoning_element.props.get("isThinking"):
                reasoning_element.props = {
                    "thinking": thinking_text,
                    "isThinking": False,
                    "stats": stats_data or {},
                }
                msg.elements = [reasoning_element]
                await msg.update()

            reply += delta
            await msg.stream_token(delta)

        elif event_type == "stats":
            stats_data = event.get("data") or {}
            reasoning_element.props = {
                "thinking": thinking_text,
                "isThinking": False,
                "stats": stats_data,
            }
            msg.elements = [reasoning_element]
            await msg.update()

    try:
        logger.info(f"Calling assistant chat API with model: {selected_model}")
        await stream_chat_events_callback(selected_model, history, on_assistant_event)

    except Exception as e:
        logger.exception(f"Assistant chat failed: {e}")
        reply = f"⚠ Assistant chat failed: {str(e)}"
        await msg.stream_token(reply)

    reasoning_element.props = {
        "thinking": thinking_text,
        "isThinking": False,
        "stats": stats_data or {},
    }
    msg.elements = [reasoning_element]
    await msg.update()
    await msg.update()

    history.append({"role": "assistant", "content": reply})
    cl.user_session.set("history", history)
    append_event("assistant", reply)

    if stats_data and stats_data.get("tok_s"):
        logger.info(f"Assistant generation speed: {stats_data['tok_s']:.2f} tok/s")



async def start_assistant_session(username: str | None, role: str | None):
    """Initialize a normal Assistant chat.

    Assistant mode intentionally has no char_id and no character singleton memory.
    Each Chainlit chat/session remains independent.
    """
    sid = cl.user_session.get("id") or "unknown"
    external_user_id = f"chainlit:username:{username}" if username else f"chainlit:session:{sid}"

    cl.user_session.set("mode", "assistant")
    cl.user_session.set("char", None)
    cl.user_session.set("char_id", None)
    cl.user_session.set("external_user_id", external_user_id)

    logger.info(f"[IDENT] assistant user={username!r} external_user_id={external_user_id!r}")

    ident = await identity_ensure(external_user_id, username)
    if not ident:
        ident = {"person_id": f"local:{username or sid}", "preferred_name": username or "there"}

    cl.user_session.set("person_id", ident.get("person_id"))
    preferred_name = ident.get("preferred_name") or username or "there"
    cl.user_session.set("preferred_name", preferred_name)

    cl.user_session.set(
        "history",
        [{"role": "system", "content": assistant_system_text(preferred_name)}],
    )

    # Pick a usable default chat model for Assistant mode.
    try:
        model_names = await fetch_models()
        chat_models = [m for m in model_names if not is_embedding_model(m)]
        default_model = choose_default_chat_model(model_names)

        cl.user_session.set("model_available", default_model is not None)

        if default_model:
            cl.user_session.set("default_model", default_model)
            cl.user_session.set("model", default_model)
            cl.user_session.set("selected_model", default_model)
            logger.info(f"Assistant started with default chat model: {default_model}")
            await send_model_settings(chat_models, default_model)
        else:
            logger.warning("Assistant started, but no chat models were returned by MODELS_API_URL")
    except Exception as e:
        logger.error(f"Assistant model fetch failed: {e}")
        cl.user_session.set("model_available", False)

    await cl.Message(
        content="Assistant mode ready. This chat is independent from character chats."
    ).send()


class LocalPublicStorageClient(BaseStorageClient):
    """Local Chainlit storage client.

    Stores uploaded Chainlit element/file payloads under /app/public/chainlit_storage
    and returns URLs served by Chainlit's public directory.
    """

    def __init__(self):
        self.root = Path(os.getenv("CHAINLIT_LOCAL_STORAGE_DIR", "/app/public/chainlit_storage")).resolve()
        self.public_prefix = os.getenv("CHAINLIT_LOCAL_STORAGE_URL_PREFIX", "/public/chainlit_storage").rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, object_key: str) -> tuple[str, Path]:
        key = str(object_key or "").lstrip("/").replace("\\", "/")
        if not key:
            key = "unnamed"

        target = (self.root / key).resolve()

        try:
            target.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Unsafe storage object key: {object_key!r}")

        return key, target

    async def upload_file(
        self,
        object_key: str,
        data,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ):
        key, target = self._safe_path(object_key)

        if target.exists() and not overwrite:
            return {
                "object_key": key,
                "url": await self.get_read_url(key),
            }

        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = data

        target.write_bytes(payload)

        return {
            "object_key": key,
            "url": await self.get_read_url(key),
        }

    async def delete_file(self, object_key: str) -> bool:
        _, target = self._safe_path(object_key)
        if target.exists():
            target.unlink()
            return True
        return False

    async def get_read_url(self, object_key: str) -> str:
        key, _ = self._safe_path(object_key)
        return f"{self.public_prefix}/{quote(key, safe='/')}"

    async def close(self) -> None:
        return None


@cl.data_layer
def get_data_layer():
    if ChainlitDataLayer is None:
        logger.warning("ChainlitDataLayer unavailable; returning no data layer")
        return None
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    return ChainlitDataLayer(
        database_url=database_url,
        storage_client=LocalPublicStorageClient(),
        show_logger=False,
    )



@cl.set_chat_profiles
async def chat_profiles():
    """Define available chat profiles.

    Assistant is created directly in Chainlit.
    Real characters are loaded from the character API.
    """
    global CHAR_INDEX, CHAR_LIST, PROFILE_NAME_TO_ID

    profiles = [
        cl.ChatProfile(
            name=ASSISTANT_PROFILE_NAME,
            icon="",
            markdown_description="General assistant mode with independent chats. Not a roleplay character.",
            starters=[
                cl.Starter(
                    label="Start assistant chat",
                    message="Help me with "
                )
            ],
        )
    ]

    try:
        CHAR_LIST = await fetch_characters_list()
    except Exception as e:
        logger.warning(f"Failed to fetch characters from API: {e}, using cache")
        CHAR_LIST = load_cached_characters()

    # Remove old Assistant pseudo-character records from the character API.
    # Assistant is now a first-class Chainlit mode.
    CHAR_LIST = [char for char in CHAR_LIST if not is_assistant_character_record(char)]

    CHAR_INDEX = {char["id"]: char for char in CHAR_LIST if "id" in char}
    PROFILE_NAME_TO_ID = {
        char["name"]: char["id"]
        for char in CHAR_LIST
        if "id" in char and "name" in char
    }

    if not CHAR_LIST:
        logger.warning("No character profiles available; only Assistant profile will be shown")
        return profiles

    for char in CHAR_LIST:
        try:
            icon = char.get("avatar_url") or (
                f"{CHAR_API_URL}/avatars/{char.get('avatar', '')}"
                if CHAR_API_URL else ""
            )

            profiles.append(
                cl.ChatProfile(
                    name=char["name"],
                    icon=icon,
                    markdown_description=char.get("description", ""),
                    starters=[
                        cl.Starter(
                            label="Continue conversation",
                            message=char.get("greeting", "Hello there!")
                        )
                    ],
                )
            )
        except Exception as e:
            logger.error(f"Failed to create profile for {char.get('id')}: {e}")

    return profiles

@cl.on_chat_start
async def start():
    """Initialize chat session with selected character and model settings.
    
    Sets up user session with character data, conversation history,
    and model selection dropdown in sidebar.
    """
    global CHAR_LIST, CHAR_INDEX, PROFILE_NAME_TO_ID

    # Pull authenticated user from Chainlit session (context exists here)
    u = cl.user_session.get("user")
    username = getattr(u, "identifier", None) if u else None
    role = (getattr(u, "metadata", {}) or {}).get("role") if u else None

    cl.user_session.set("auth_username", username)
    cl.user_session.set("auth_role", role)

    if not CHAR_LIST:
        try:
            CHAR_LIST = await fetch_characters_list()
        except Exception as e:
            logger.warning(f"Failed to fetch characters from API in start(): {e}, using cache")
            CHAR_LIST = load_cached_characters()
        CHAR_INDEX = {c["id"]: c for c in CHAR_LIST} if CHAR_LIST else {}
        PROFILE_NAME_TO_ID = {
            c["name"]: c["id"]
            for c in CHAR_LIST
            if "id" in c and "name" in c
        }

    current_profile_name = cl.user_session.get("chat_profile")

    if current_profile_name == ASSISTANT_PROFILE_NAME:
        await start_assistant_session(username, role)
        return

    if not CHAR_LIST:
        await cl.Message(content="⚠️ No characters loaded. Please check configuration.").send()
        return

    char_id = PROFILE_NAME_TO_ID.get(current_profile_name)
    
    if not char_id:
        char_id = CHAR_LIST[0]["id"]
        logger.warning(f"Profile '{current_profile_name}' not found, using {CHAR_LIST[0]['name']}")

    try:
        char = await fetch_character_private(char_id)
        logger.info(f"Fetched full character data for: {char['name']}")
        logger.info("CHAR UI DEBUG name=%r avatar=%r avatar_url=%r", char.get("name"), char.get("avatar"), char.get("avatar_url"))
    except Exception as e:
        logger.error(f"Failed to fetch character details: {e}")
        await cl.Message(content="⚠️ Failed to load character. Please try again.").send()
        return

    cl.user_session.set("mode", "character")
    cl.user_session.set("char", char)
    cl.user_session.set("char_id", char_id)
    
    # Register character avatar for message author
    await register_character_avatar(char)
    
    # Resolve user identity (reliable)
    username = cl.user_session.get("auth_username")
    if not username:
        app_user = getattr(cl, "user", None)
        username = getattr(app_user, "identifier", None) if app_user else None
    
    # Fallback: if somehow missing, use session id (still stable per chat)
    sid = cl.user_session.get("id") or "unknown"
    external_user_id = f"chainlit:username:{username}" if username else f"chainlit:session:{sid}"
    
    cl.user_session.set("external_user_id", external_user_id)
    logger.info(f"[IDENT] user={username!r} external_user_id={external_user_id!r}")
    
    ident = await identity_ensure(external_user_id, username)
    if not ident:
        ident = {"person_id": f"local:{username or sid}", "preferred_name": username or "there"}
    cl.user_session.set("person_id", ident.get("person_id"))
    preferred_name = ident.get("preferred_name") or username or "there"
    cl.user_session.set("preferred_name", preferred_name)
    
    # Inject identity hint into system prompt
    identity_hint = (
        f"You are chatting with a user whose preferred name is '{preferred_name}'. "
        f"Always address them as '{preferred_name}' unless they explicitly ask otherwise. "
        f"Do not ask them what their name is - you already know it.\n\n"
    )
    system_text = (char.get("prompts") or {}).get("system") or ""
    cl.user_session.set("history", [{"role": "system", "content": identity_hint + system_text}])
    logger.info(f"Chat started with character: {char['name']} for user: {username} (preferred: {preferred_name})")

    # Model selection sidebar with error handling
    try:
        model_names = await fetch_models()
        
        if model_names:
            default_model = choose_default_chat_model(model_names)
            model_names = [m for m in model_names if not is_embedding_model(m)]
            if default_model:
                logger.info(f"Found {len(model_names)} chat models, using {default_model} as default")
            else:
                logger.warning("No chat models available")
                model_names = ["No models available"]
        else:
            logger.warning("No models available")
            model_names = ["No models available"]
            default_model = None
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        model_names = ["No models available"]
        default_model = None

    # Store whether models are available
    cl.user_session.set("model_available", default_model is not None)
    if default_model:
        cl.user_session.set("default_model", default_model)

    await send_model_settings(model_names, default_model or model_names[0], "Auto")
    
    # Log session start
    append_event("system", f"session_start character={char_id}")
    
    # Send character greeting
    greeting = char.get("greeting")
    if greeting:
        await send_character_message(greeting, char)


async def fetch_assistant_thread_history_from_db_v2(thread_id: str, preferred_name: str):
    """Rebuild Assistant model history from persisted Chainlit Step rows."""
    history = [{"role": "system", "content": assistant_system_text(preferred_name)}]

    database_url = os.getenv("DATABASE_URL")
    if not database_url or not thread_id:
        return history

    query = (
        'SELECT '
        '"type", "name", "input", "output", "createdAt", "startTime", "id" '
        'FROM "Step" '
        'WHERE "threadId" = $1::uuid '
        'ORDER BY COALESCE("createdAt", "startTime") ASC NULLS LAST, "id" ASC'
    )

    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(query, str(thread_id))
    finally:
        await conn.close()

    for row in rows:
        step_type = str(row["type"] or "").lower()
        name = str(row["name"] or "").lower()

        content = row["output"] or row["input"] or ""
        if not isinstance(content, str):
            continue

        content = content.strip()
        if not content:
            continue

        if "Assistant mode ready. This chat is independent from character chats." in content:
            continue
        if step_type == "run":
            continue
        if name in {"reasoningpanel", "thinking"}:
            continue

        if step_type == "user_message":
            history.append({"role": "user", "content": content})
        elif step_type == "assistant_message":
            history.append({"role": "assistant", "content": content})

    return history



@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume a persisted Assistant thread with real model history."""
    thread_id = None
    if isinstance(thread, dict):
        thread_id = thread.get("id") or thread.get("threadId") or thread.get("thread_id")
    else:
        thread_id = getattr(thread, "id", None) or getattr(thread, "threadId", None) or getattr(thread, "thread_id", None)

    if thread_id:
        thread_id = str(thread_id)
        cl.user_session.set("resumed_thread_id", thread_id)
        cl.user_session.set("loaded_thread_id", thread_id)
        cl.user_session.set("current_thread_id", thread_id)

    u = cl.user_session.get("user")
    username = getattr(u, "identifier", None) if u else cl.user_session.get("auth_username")
    role = (getattr(u, "metadata", {}) or {}).get("role") if u else cl.user_session.get("auth_role")

    sid = cl.user_session.get("id") or "unknown"
    external_user_id = f"chainlit:username:{username}" if username else f"chainlit:session:{sid}"

    cl.user_session.set("mode", "assistant")
    cl.user_session.set("chat_profile", ASSISTANT_PROFILE_NAME)
    cl.user_session.set("char", None)
    cl.user_session.set("char_id", None)
    cl.user_session.set("auth_username", username)
    cl.user_session.set("auth_role", role)
    cl.user_session.set("external_user_id", external_user_id)

    ident = await identity_ensure(external_user_id, username)
    if not ident:
        ident = {"person_id": f"local:{username or sid}", "preferred_name": username or "there"}

    cl.user_session.set("person_id", ident.get("person_id"))
    preferred_name = ident.get("preferred_name") or username or "there"
    cl.user_session.set("preferred_name", preferred_name)

    model_names = await fetch_models()
    model_names = [m for m in model_names if not is_embedding_model(m)]
    default_model = choose_default_chat_model(model_names)

    cl.user_session.set("model_available", bool(default_model))
    if default_model:
        cl.user_session.set("default_model", default_model)

    settings = cl.user_session.get("settings") or {}
    if default_model and not settings.get("Model"):
        settings["Model"] = default_model
    settings.setdefault("Thinking", "Auto")
    cl.user_session.set("settings", settings)

    if model_names:
        await send_model_settings(model_names, settings.get("Model") or default_model, settings.get("Thinking", "Auto"))

    if thread_id:
        history = await fetch_assistant_thread_history_from_db_v2(thread_id, preferred_name)
    else:
        history = [{"role": "system", "content": assistant_system_text(preferred_name)}]

    cl.user_session.set("history", history)

    logger.info(
        f"Resumed Assistant thread {thread_id!r} with {max(len(history)-1, 0)} model history messages"
    )


@cl.on_settings_update
async def update_settings(settings):
    """Handle model/thinking setting changes from sidebar settings."""
    settings.setdefault("Thinking", "Auto")
    cl.user_session.set("settings", settings)
    logger.info(f"Settings updated: Model={settings.get('Model')!r} Thinking={settings.get('Thinking')!r}")

@cl.on_message
async def main(message: cl.Message):
    """Process incoming user messages and generate AI responses.
    
    Handles message streaming, emotion detection, and conversation history.
    
    :param message: Incoming message from user
    :type message: cl.Message
    """
    
    # Debug command: /whoami
    if message.content.strip().lower() == "/whoami":
        username = cl.user_session.get("auth_username")
        role = cl.user_session.get("auth_role")
        
        # Fallback to chainlit user object if not in session
        if not username or not role:
            u = cl.user_session.get("user")
            if u:
                username = username or getattr(u, "identifier", None)
                role = role or (getattr(u, "metadata", {}) or {}).get("role")
        
        await cl.Message(
            content=(
                f"username: {username}\n"
                f"role: {role}\n"
                f"external_user_id: {cl.user_session.get('external_user_id')}\n"
                f"person_id: {cl.user_session.get('person_id')}\n"
                f"preferred_name: {cl.user_session.get('preferred_name')}"
            )
        ).send()
        return
    
    # Admin command: /admin_users
    if message.content.strip() == "/admin_users":
        if not await require_admin_or_warn():
            return
        try:
            async with httpx.AsyncClient(timeout=AUTH_TIMEOUT) as c:
                r = await c.get(f"{AUTH_API_URL}/auth/admin/users", headers=_admin_headers())
            if r.status_code != 200:
                await cl.Message(content=f"⚠️ Auth API error: {r.status_code} {r.text[:200]}").send()
                return
            users = r.json().get("users", [])
            lines = [f"- {u['username']} ({u['role']}) active={u['is_active']}" for u in users]
            await cl.Message(content="Users:\n" + "\n".join(lines)).send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error: {str(e)}").send()
        return
    
    # Admin command: /admin_invite [hours]
    if message.content.strip().startswith("/admin_invite"):
        if not await require_admin_or_warn():
            return
        parts = message.content.split()
        expires = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        try:
            async with httpx.AsyncClient(timeout=AUTH_TIMEOUT) as c:
                r = await c.post(f"{AUTH_API_URL}/auth/admin/invite", json={"expires_hours": expires}, headers=_admin_headers())
            if r.status_code != 200:
                await cl.Message(content=f"⚠️ Auth API error: {r.status_code} {r.text[:200]}").send()
                return
            code = r.json().get("code")
            await cl.Message(content=f"✅ Invite code: `{code}`").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error: {str(e)}").send()
        return
    
    # Admin command: /admin_setrole <username> <role>
    if message.content.strip().startswith("/admin_setrole"):
        if not await require_admin_or_warn():
            return
        parts = message.content.split()
        if len(parts) != 3:
            await cl.Message(content="Usage: /admin_setrole <username> <admin|user>").send()
            return
        username, role = parts[1], parts[2]
        try:
            async with httpx.AsyncClient(timeout=AUTH_TIMEOUT) as c:
                r = await c.post(f"{AUTH_API_URL}/auth/admin/set_role", json={"username": username, "role": role}, headers=_admin_headers())
            if r.status_code != 200:
                await cl.Message(content=f"⚠️ Auth API error: {r.status_code} {r.text[:200]}").send()
                return
            msg = r.json().get("message")
            await cl.Message(content=f"✅ {msg}").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error: {str(e)}").send()
        return
    
    # Admin command: /admin_disable <username>
    if message.content.strip().startswith("/admin_disable"):
        if not await require_admin_or_warn():
            return
        parts = message.content.split()
        if len(parts) != 2:
            await cl.Message(content="Usage: /admin_disable <username>").send()
            return
        username = parts[1]
        try:
            async with httpx.AsyncClient(timeout=AUTH_TIMEOUT) as c:
                r = await c.post(f"{AUTH_API_URL}/auth/admin/disable", json={"username": username}, headers=_admin_headers())
            if r.status_code != 200:
                await cl.Message(content=f"⚠️ Auth API error: {r.status_code} {r.text[:200]}").send()
                return
            msg = r.json().get("message")
            await cl.Message(content=f"✅ {msg}").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error: {str(e)}").send()
        return
    
    # Admin command: /admin_enable <username>
    if message.content.strip().startswith("/admin_enable"):
        if not await require_admin_or_warn():
            return
        parts = message.content.split()
        if len(parts) != 2:
            await cl.Message(content="Usage: /admin_enable <username>").send()
            return
        username = parts[1]
        try:
            async with httpx.AsyncClient(timeout=AUTH_TIMEOUT) as c:
                r = await c.post(f"{AUTH_API_URL}/auth/admin/enable", json={"username": username}, headers=_admin_headers())
            if r.status_code != 200:
                await cl.Message(content=f"⚠️ Auth API error: {r.status_code} {r.text[:200]}").send()
                return
            msg = r.json().get("message")
            await cl.Message(content=f"✅ {msg}").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Error: {str(e)}").send()
        return

    mode = cl.user_session.get("mode")
    if mode == "assistant":
        await handle_assistant_message(message)
        return

    char = cl.user_session.get("char")
    if not char:
        await cl.Message(content="⚠️ No character selected. Please restart the chat.").send()
        return

    model_available = cl.user_session.get("model_available", False)
    if not model_available:
        await send_character_message("⚠️ No models available. Please check API configuration.", char)
        return

    settings = cl.user_session.get("settings", {})
    default_model = cl.user_session.get("default_model")
    selected_model = settings.get("Model", default_model)

    history = cl.user_session.get("history")
    if not history:
        preferred_name = cl.user_session.get("preferred_name", "there")
        identity_hint = (
            f"You are chatting with a user whose preferred name is '{preferred_name}'. "
            f"Always address them as '{preferred_name}' unless they explicitly ask otherwise. "
            f"Do not ask them what their name is - you already know it.\n\n"
        )
        history = [{"role": "system", "content": identity_hint + char.get("prompts", {}).get("system", "")}]
    history.append({"role": "user", "content": message.content})
    append_event("user", message.content)

    reply = ""
    stats_data = None
    thinking_text = ""
    last_reasoning_update = 0.0
    reasoning_update_interval = 0.25

    msg = cl.Message(content="", author=character_display_name(char))
    await msg.send()

    reasoning_element = cl.CustomElement(
        name="ReasoningPanel",
        props={
            "thinking": "",
            "isThinking": True,
            "stats": {},
        },
        display="inline",
    )

    msg.elements = [reasoning_element]
    await msg.update()

    try:
        logger.info(f"Calling chat API with model: {selected_model}")

        async for event in stream_chat_events(selected_model, history):
            event_type = event.get("type")

            if event_type == "thinking_delta":
                delta = event.get("content") or ""
                if not delta:
                    continue

                thinking_text += delta

                now = time.monotonic()
                if now - last_reasoning_update >= reasoning_update_interval:
                    reasoning_element.props = {
                        "thinking": thinking_text,
                        "isThinking": True,
                        "stats": stats_data or {},
                    }
                    msg.elements = [reasoning_element]
                    await msg.update()
                    last_reasoning_update = now

            elif event_type == "answer_delta":
                delta = event.get("content") or ""
                if not delta:
                    continue

                if thinking_text and reasoning_element.props.get("isThinking"):
                    reasoning_element.props = {
                        "thinking": thinking_text,
                        "isThinking": False,
                        "stats": stats_data or {},
                    }
                    msg.elements = [reasoning_element]
                    await msg.update()

                reply += delta
                await msg.stream_token(delta)

            elif event_type == "stats":
                stats_data = event.get("data") or {}
                reasoning_element.props = {
                    "thinking": thinking_text,
                    "isThinking": False,
                    "stats": stats_data,
                }
                msg.elements = [reasoning_element]
                await msg.update()

            elif event_type == "done":
                # Do not break here. Let stream_chat_events() finish naturally
                # so httpx can close the streaming response cleanly.
                continue
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        reply = f"⚠ Chat API error: {str(e)}"
        await msg.stream_token(reply)

    reasoning_element.props = {
        "thinking": thinking_text,
        "isThinking": False,
        "stats": stats_data or {},
    }
    msg.elements = [reasoning_element]
    await msg.update()

    # Emotion detection with error handling
    if reply.strip() and EMOTION_ENABLED:
        emotion_result = await detect_emotion(reply)
        if emotion_result and emotion_result.get("label"):
            await cl.Message(
                content=f"Emotion: {emotion_result['label'].capitalize()} (confidence: {emotion_result['score']:.2f})",
                disable_human_feedback=True
            ).send()

    history.append({"role": "assistant", "content": reply})
    cl.user_session.set("history", history)
    append_event("assistant", reply)

@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when chat session ends.
    
    Logs session end event for persistent conversation tracking.
    """
    append_event("system", "session_end")

@cl.action_callback("heartbeat")
async def heartbeat():
    """Handle heartbeat action to maintain activity status.
    
    :return: Status string indicating active state
    :rtype: str
    """
    return "Active"

# Shutdown hook for clean container stop
import atexit
import asyncio
import signal

def _close_httpx_sync():
    """Best-effort close for the global AsyncClient on interpreter exit."""
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None

    if loop and loop.is_running():
        # Schedule and hope the loop still runs long enough
        try:
            loop.create_task(client.aclose())
        except Exception:
            pass
        return

    # No running loop: create one just to close
    try:
        asyncio.run(client.aclose())
    except Exception:
        pass

def _handle_sigterm(*_):
    _close_httpx_sync()

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)
atexit.register(_close_httpx_sync)
