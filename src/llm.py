import os
import time

# Запросы к localhost (Ollama) не должны идти через VPN/прокси, иначе приходит пустой 503
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# Выбор режима работы: если задан ключ Groq, используем облачные модели,
# иначе локальные через Ollama
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    # Две модели разного размера и стоимости (цена в $ за 1 млн токенов):
    # small - для служебных шагов графа, large - для итогового ответа
    MODELS = {
        "small": dict(name="llama-3.1-8b-instant", pin=0.05, pout=0.08),
        "large": dict(name="llama-3.3-70b-versatile", pin=0.59, pout=0.79),
    }

    def _make(name):
        return ChatGroq(model=name, temperature=0.2)
else:
    from langchain_ollama import ChatOllama
    # Локальные модели бесплатны, поэтому цены условные:
    # они нужны, чтобы Cost Dashboard показывал разные затраты у моделей разных размеров
    MODELS = {
        "small": dict(name="llama3.2:3b", pin=0.02, pout=0.02),     # быстрая, служебные шаги
        "large": dict(name="llama3.1:latest", pin=0.10, pout=0.10), # 8B, финальный ответ
    }

    def _make(name):
        # 127.0.0.1 - адрес Ollama на этом компьютере
        # temperature=0.2 - меньше фантазии, ответ ближе к контексту
        # num_ctx=4096 - размер окна контекста (влияет на память)
        # num_predict=400 - максимум токенов в ответе, чтобы генерация не затягивалась
        return ChatOllama(model=name, base_url="http://127.0.0.1:11434",
                          temperature=0.2, num_ctx=4096, num_predict=400)

_cache = {}   # создаём каждую модель один раз и переиспользуем


def call(tier, node, prompt):
    """Вызов LLM. Возвращает (текст, запись для Cost Dashboard).
    tier - уровень модели ("small" или "large"), node - имя узла графа, откуда вызвали."""
    cfg = MODELS[tier]
    if tier not in _cache:
        _cache[tier] = _make(cfg["name"])
    t = time.time()
    resp = _cache[tier].invoke(prompt)
    sec = time.time() - t                       # сколько секунд ушло на вызов
    text = resp.content.strip()
    # Если модель сообщила число токенов, берём его; иначе оцениваем по длине текста
    # (грубо: 1 токен ~ 3 символа)
    um = getattr(resp, "usage_metadata", None) or {}
    tin = um.get("input_tokens") or len(prompt) // 3
    tout = um.get("output_tokens") or len(text) // 3
    return text, record(node, cfg["name"], tin, tout, sec, cfg["pin"], cfg["pout"])


def record(node, model, tin=0, tout=0, sec=0.0, pin=0.0, pout=0.0):
    """Запись о затратах одного шага: узел, модель, токены, время, стоимость.
    Стоимость = (токены на входе * цена входа + токены на выходе * цена выхода) / 1 млн.
    Используется и для шагов без LLM (поиск, спеллчекер, веб): у них токены и цена нулевые."""
    return dict(node=node, model=model, tokens_in=tin, tokens_out=tout,
                sec=round(sec, 2), cost=(tin * pin + tout * pout) / 1e6)
