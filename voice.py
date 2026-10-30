"""Voice: speech-to-text and text-to-speech (gTTS, Hindi or English)."""
import base64
import io
import re

from app import config
from app.rag import generator


def transcribe(audio: bytes, filename: str = "audio.wav") -> str:
    """Gemini: audio is sent to the chat model. Groq: Whisper endpoint."""
    client = generator.client()
    if config.LLM_PROVIDER == "groq":
        result = client.audio.transcriptions.create(file=(filename, audio), model=config.WHISPER_MODEL)
        return (result.text or "").strip()
    fmt = "mp3" if filename.lower().endswith(".mp3") else "wav"
    resp = client.chat.completions.create(model=config.LLM_MODEL, max_tokens=800, messages=[{
        "role": "user", "content": [
            {"type": "text", "text": "Transcribe this audio exactly. The speaker may use English, Hindi or "
                                     "Hinglish. Output only the transcript."},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": fmt}}]}])
    return (resp.choices[0].message.content or "").strip()


def speak(text: str) -> bytes:
    from gtts import gTTS
    clean = re.sub(r"\[\d+\]|[*_`#>]", "", text)
    lang = "hi" if re.search(r"[\u0900-\u097F]", clean) else "en"
    buf = io.BytesIO()
    gTTS(clean[:1500], lang=lang).write_to_fp(buf)
    return buf.getvalue()
