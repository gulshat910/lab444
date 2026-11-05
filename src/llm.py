import os
import time

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    MODELS = {
        "small": dict(name="llama-3.1-8b-instant", pin=0.05, pout=0.08),
        "large": dict(name="llama-3.3-70b-versatile", pin=0.59, pout=0.79),
    }

    def _make(name):
        return ChatGroq(model=name, temperature=0.2)
else:
    from langchain_ollama import ChatOllama
    # локальные модели бесплатны, цены условные ($ за 1M токенов) для Cost Dashboard
    MODELS = {
        "small": dict(name="llama3.2:3b", pin=0.02, pout=0.02),
        "large": dict(name="llama3.1:latest", pin=0.10, pout=0.10),
    }

    def _make(name):
        return ChatOllama(model=name, base_url="http://127.0.0.1:11434",
                          temperature=0.2, num_ctx=4096, num_predict=400)

_cache = {}


def call(tier, node, prompt):
    """Вызов LLM. Возвращает (текст, запись для Cost Dashboard)."""
    cfg = MODELS[tier]
    if tier not in _cache:
        _cache[tier] = _make(cfg["name"])
    t = time.time()
    resp = _cache[tier].invoke(prompt)
    sec = time.time() - t
    text = resp.content.strip()
    um = getattr(resp, "usage_metadata", None) or {}
    tin = um.get("input_tokens") or len(prompt) // 3
    tout = um.get("output_tokens") or len(text) // 3
    return text, record(node, cfg["name"], tin, tout, sec, cfg["pin"], cfg["pout"])


def record(node, model, tin=0, tout=0, sec=0.0, pin=0.0, pout=0.0):
    return dict(node=node, model=model, tokens_in=tin, tokens_out=tout,
                sec=round(sec, 2), cost=(tin * pin + tout * pout) / 1e6)
