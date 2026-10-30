"""Password hashing (PBKDF2) and minimal HS256 JWT, using only the standard library."""
import base64
import hashlib
import hmac
import json
import os
import time


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    salt = stored.split("$", 1)[0]
    return hmac.compare_digest(hash_password(password, salt), stored)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(message: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest())


def create_token(payload: dict, secret: str, ttl_hours: int = 12) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({**payload, "exp": int(time.time() + ttl_hours * 3600)}).encode())
    return f"{header}.{body}.{_sign(f'{header}.{body}', secret)}"


def decode_token(token: str, secret: str) -> dict | None:
    try:
        header, body, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(f"{header}.{body}", secret)):
            return None
        payload = json.loads(_unb64(body))
        return payload if payload.get("exp", 0) >= time.time() else None
    except Exception:
        return None
