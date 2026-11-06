import os
import re
import sqlite3
import time
from functools import lru_cache

import faiss
import numpy as np
import requests
from langchain_huggingface import HuggingFaceEmbeddings

# Базы лабы 4: тексты и FTS5-индекс (lab4.db), векторы (lab4.faiss)
DB, IDX = "data/lab4.db", "data/lab4.faiss"
OLD_DB, OLD_IDX = "data/metadata.db", "data/faiss.index"   # архив Хабра из лабы 2

# Какие колонки читаем из таблицы metadata
COLS = "id, source, title, url, date, topic, chunk_id, text"
KEYS = COLS.split(", ")
# Стоп-слова: частые слова, которые не должны влиять на поиск по словам
STOP = {"что", "как", "какой", "какая", "какие", "какое", "это", "для", "при", "или",
        "его", "она", "они", "был", "была", "были", "кто", "где", "когда", "сколько",
        "через", "про", "над", "под", "все", "так", "какую", "the", "and", "for"}


@lru_cache(maxsize=1)
def _res():
    """Загружает модель эмбеддингов и индекс FAISS один раз (кэш), дальше переиспользует.
    Поэтому первый запрос долгий, следующие быстрее."""
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    return emb, faiss.read_index(IDX)


@lru_cache(maxsize=1)
def _old_index():
    """Векторный индекс старого архива Хабра (тоже загружается один раз)."""
    return faiss.read_index(OLD_IDX)


def _db():
    return sqlite3.connect(DB, check_same_thread=False)


def _match(q):
    """Запрос -> FTS5: русские варианты заменяем английскими терминами,
    берём префиксы из 6 букв (грубая морфология), убираем стоп-слова.
    Результат вида: "python"* OR "async"*"""
    q = q.lower()
    q = re.sub(r"питон\w*", "python", q)          # в статьях пишут python, а не «питон»
    q = re.sub(r"асинхрон\w*", "async asyncio", q)
    words = [w for w in re.findall(r"\w+", q) if len(w) >= 3 and w not in STOP]
    # первые 6 букв + звёздочка: «компании» и «компания» находят друг друга
    return " OR ".join(f'"{t}"*' for t in dict.fromkeys(w[:6] for w in words))


def _old_search(query, qv):
    """Архив Хабра из лабы 2: возвращает два списка - по смыслу (FAISS) и по словам (BM25)."""
    db = sqlite3.connect(OLD_DB, check_same_thread=False)
    # векторный поиск: 20 ближайших фрагментов, FAISS возвращает их id
    _, ids = _old_index().search(qv, 20)
    ids = [int(i) for i in ids[0] if i != -1]
    vec = []
    if ids:
        # по id достаём тексты и метаданные из SQLite
        rows = {r[0]: r for r in db.execute(
            "SELECT id,title,url,published,chunk_id,text FROM metadata "
            f"WHERE id IN ({','.join('?' * len(ids))})", ids)}
        vec = [(rows[i], "vector") for i in ids if i in rows]
    # поиск по словам: BM25, заголовок весит втрое больше текста
    bm, m = [], _match(query)
    if m:
        bm = [(r, "bm25") for r in db.execute(
            "SELECT m.id,m.title,m.url,m.published,m.chunk_id,m.text FROM fts "
            "JOIN metadata m ON m.id=fts.rowid WHERE fts MATCH ? "
            "ORDER BY bm25(fts,3.0,1.0) LIMIT 20", (m,))]
    db.close()

    def conv(lst):
        # приводим строки старой базы к тому же виду, что и у lab4.db
        # (uid с префиксом "o", чтобы id не совпали с новой базой)
        return [(dict(uid=f"o{r[0]}", source="habr", title=r[1], url=r[2], date=r[3],
                      topic="архив Хабра", chunk_id=r[4], text=r[5]), t) for r, t in lst]

    return conv(vec), conv(bm)


def search(query, source, k=5):
    """Гибридный поиск (FAISS + BM25 FTS5, RRF) внутри источника (habr / news / science)."""
    emb, index = _res()
    # вопрос превращается в вектор (эмбеддинг)
    qv = np.array([emb.embed_query(query)], dtype="float32")
    db = _db()

    def new(lst):
        return [(dict(zip(KEYS, r), uid=str(r[0])), t) for r, t in lst]

    # 1) Поиск по смыслу: FAISS отдаёт 300 ближайших id (с запасом),
    #    из них оставляем только фрагменты нужного источника, берём 20 лучших
    _, ids = index.search(qv, 300)
    ids = [int(i) for i in ids[0] if i != -1]
    rows = {r[0]: r for r in db.execute(
        f"SELECT {COLS} FROM metadata WHERE source=? AND id IN ({','.join('?' * len(ids))})",
        [source] + ids)}
    lists = [new([(rows[i], "vector") for i in ids if i in rows][:20])]

    # 2) Поиск по словам: FTS5 + BM25 внутри того же источника, 20 лучших
    m = _match(query)
    if m:
        cols = ", ".join("m." + c for c in KEYS)
        r = db.execute(f"SELECT {cols} FROM fts JOIN metadata m ON m.id=fts.rowid "
                       "WHERE fts MATCH ? AND m.source=? ORDER BY bm25(fts, 3.0, 1.0) LIMIT 20",
                       (m, source)).fetchall()
        lists.append(new([(x, "bm25") for x in r]))
    db.close()

    # 3) Для Хабра добавляем списки из старого архива (лаба 2)
    if source == "habr" and os.path.exists(OLD_DB) and os.path.exists(OLD_IDX):
        lists.extend(_old_search(query, qv))

    # 4) Reciprocal Rank Fusion: фрагмент получает 1/(60 + место) за каждый список,
    #    баллы суммируются; побеждают фрагменты, высоко стоящие в нескольких списках
    score, keep = {}, {}
    for lst in lists:
        for rank, (d, typ) in enumerate(lst, 1):
            score[d["uid"]] = score.get(d["uid"], 0) + 1 / (60 + rank)
            keep.setdefault(d["uid"], dict(d, search_type=typ))
    return [keep[u] for u in sorted(score, key=score.get, reverse=True)[:k]]


def fix_typos(text):
    """Яндекс.Спеллер (пункт 3). Слова с заглавной буквы (имена) не трогаем.
    Возвращает (исправленный текст, число исправлений, секунды)."""
    t = time.time()
    try:
        r = requests.get("https://speller.yandex.net/services/spellservice.json/checkText",
                         params={"text": text, "lang": "ru,en"}, timeout=8).json()
        n = 0
        # идём с конца текста, чтобы замены не сдвигали позиции остальных ошибок
        for e in sorted(r, key=lambda e: -e["pos"]):
            word = text[e["pos"]:e["pos"] + e["len"]]
            if e.get("s") and not word[:1].isupper():
                text = text[:e["pos"]] + e["s"][0] + text[e["pos"] + e["len"]:]
                n += 1
        return text, n, time.time() - t
    except Exception:
        # спеллер недоступен (сеть, VPN): продолжаем с исходным текстом
        return text, 0, time.time() - t


def web_search(query, n=4):
    """Веб-поиск через DuckDuckGo (пункт 6). При ошибке возвращает пустой список."""
    try:
        from ddgs import DDGS
        res = DDGS().text(query, max_results=n, region="ru-ru")
        return [dict(title=r["title"], url=r["href"], text=r["body"]) for r in res]
    except Exception:
        return []
