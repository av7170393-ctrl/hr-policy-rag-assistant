"""LLM calls via the OpenAI-compatible API (Gemini free tier or Groq): rewriting and grounded answers."""
import os

from app import config

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        if config.LLM_PROVIDER not in config.PROVIDERS:
            raise RuntimeError("LLM_PROVIDER must be 'gemini' or 'groq'.")
        cfg = config.PROVIDERS[config.LLM_PROVIDER]
        key = os.getenv(cfg["key_env"])
        if not key:
            raise RuntimeError(f"{cfg['key_env']} is not set. Add it to your .env file.")
        _client = OpenAI(base_url=cfg["base_url"], api_key=key)
    return _client


def chat(messages, **kwargs):
    return client().chat.completions.create(model=config.LLM_MODEL, messages=messages, temperature=0, **kwargs)


def rewrite_query(question: str, history: list[tuple[str, str]]) -> str:
    """Make the question standalone and translate Hindi/Hinglish to English for retrieval."""
    convo = "\n".join(f"{role}: {text[:300]}" for role, text in history[-4:])
    prompt = ("Rewrite the employee's latest question as a short, standalone English search query for an HR "
              "policy database. Translate Hindi or Hinglish to English. Use the conversation to resolve "
              "pronouns like 'it' or 'that'. Output ONLY the query.\n\n"
              f"Conversation:\n{convo or '(none)'}\n\nLatest question: {question}")
    out = chat([{"role": "user", "content": prompt}], max_tokens=300).choices[0].message.content.strip()
    return out.strip('"\' \n') or question


SYSTEM = (
    "You are the HR Policy Assistant for Nexora Technologies. Answer ONLY from the numbered policy context. "
    "Be concise (max 5 sentences) and cite the sources you used like [1] or [2]. "
    "If the context does not contain the answer, reply with exactly: NOT_FOUND. Never invent numbers or rules. "
    "Reply in the same language and script the employee used (English, Hindi or Hinglish)."
)


def answer(question: str, chunks: list[dict], history: list[tuple[str, str]]) -> str:
    context = "\n\n".join(f"[{i}] ({c['source']} > {c['heading']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    messages = [{"role": "system", "content": SYSTEM}]
    for role, text in history[-4:]:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": f"Policy context:\n{context}\n\nQuestion: {question}"})
    return chat(messages, max_tokens=1200).choices[0].message.content.strip()
