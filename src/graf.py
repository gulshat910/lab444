import operator
import re
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.llm import call, record
from src.sources import fix_typos, search, web_search


class S(TypedDict, total=False):
    question: str
    fixed: str
    is_search: bool
    subs: List[str]
    i: int
    source: str
    query: str
    tries: int
    regen: int
    docs: list
    mode: str            # rag / web / llm
    is_ok: bool          # grade: контекст релевантен
    ok_answer: bool      # check: ответ подтверждён контекстом
    no_answer: bool      # check: ответ сообщает «в контексте нет»
    answer_i: str
    partial: list
    answer: str
    log: Annotated[list, operator.add]
    costs: Annotated[list, operator.add]


def L(node, text):
    return [dict(node=node, text=text)]


def ctx(docs, n=5, cut=700):
    return "\n\n".join(f"[{i}] {d['title']}\n{d['text'][:cut]}" for i, d in enumerate(docs[:n], 1))


CHAT_RE = re.compile(
    r"^\s*(привет|здравствуй|здорово|добрый (день|вечер|утро)|хай|hello|hi|спасибо|благодарю|"
    r"пока|до свидания|как дела|кто ты|что ты умеешь|что умеешь|ты кто)\b", re.I)

NOANS = re.compile(r"не имеет отношения|нет информации|не содержит|в контексте нет|"
                   r"не могу ответить|нет ответа|не упоминается|не говорится", re.I)


# ---------- узлы ----------

def classify(s):
    q = s["question"]
    if CHAT_RE.search(q) and len(q) < 60:
        return dict(is_search=False, costs=[],
                    log=L("classify", "разговорный запрос (правило)"))
    out, c = call("small", "classify",
        "Определи тип сообщения пользователя. Ответь ОДНИМ словом.\n"
        "SEARCH — просьба найти информацию, факты, новости, статьи, объяснить тему.\n"
        "CHAT — приветствие, благодарность, прощание, вопрос о самом помощнике, болтовня.\n\n"
        "Примеры:\nПривет! → CHAT\nСпасибо → CHAT\nЧто ты умеешь? → CHAT\n"
        "Как работает Docker? → SEARCH\nНовости про выборы → SEARCH\n\n"
        f"Сообщение: {q}\nОтвет:")
    is_search = "SEARCH" in out.upper()
    return dict(is_search=is_search, costs=[c],
                log=L("classify", "поисковый запрос" if is_search else "разговорный запрос"))


def chat(s):
    out, c = call("small", "chat",
        "Ты помощник по базе из статей Хабра, научных статей arXiv и новостей Lenta.ru. "
        "Ты умеешь искать информацию в них, исправлять опечатки, отвечать на составные "
        "вопросы и искать в интернете, если в базе ответа нет. Ответь коротко и дружелюбно "
        f"по-русски.\n\nСообщение: {s['question']}")
    return dict(answer=out, costs=[c], log=L("chat", "ответ без поиска"))


def spellcheck(s):
    fixed, n, sec = fix_typos(s["question"])
    msg = f"исправлено опечаток: {n} → «{fixed}»" if n else "опечаток не найдено"
    return dict(fixed=fixed, costs=[record("spellcheck", "Яндекс.Спеллер", sec=sec)],
                log=L("spellcheck", msg))


def decompose(s):
    q = s["fixed"]
    if not re.search(r"\bи\b.*\?|\bа также\b|\?.+\?|;", q, re.I):
        return dict(subs=[q], i=0, partial=[], costs=[],
                    log=L("decompose", "вопрос простой, разбиение не нужно"))
    out, c = call("small", "decompose",
        "Разбей вопрос на 2-3 простых вопроса, по одному на строку. Каждая строка должна "
        "быть ВОПРОСОМ. Не добавляй фактов и названий, которых нет в исходном вопросе. "
        f"Только вопросы, без нумерации.\n\nВопрос: {q}")
    subs = [re.sub(r"^[\-\d\.\)\s•]+", "", x).strip() for x in out.splitlines()]
    subs = [x for x in subs if len(x) > 5 and x.endswith("?")][:3] or [q]
    return dict(subs=subs, i=0, partial=[], costs=[c],
                log=L("decompose", f"подвопросов: {len(subs)}: " + " | ".join(subs)))


def route(s):
    q = s["subs"][s["i"]]
    out, c = call("small", "route",
        "Выбери источник для поиска ответа. Ответь ОДНИМ словом:\n"
        "habr — статьи Хабра: IT, программирование, компании, разработка, поиск, машинное обучение;\n"
        "news — новости Lenta.ru (политика, общество, культура, спорт, происшествия);\n"
        "science — научные статьи arXiv (физика, математика, информатика).\n\n"
        f"Вопрос: {q}")
    low = out.lower()
    source = next((k for k in ("habr", "news", "science") if k in low), "habr")
    costs, query = [c], q
    if source == "science":      # аннотации arXiv на английском
        query, c2 = call("small", "translate",
                         f"Переведи поисковый запрос на английский, только перевод:\n{q}")
        costs.append(c2)
    return dict(source=source, query=query, tries=0, regen=0, costs=costs,
                log=L("route", f"подвопрос «{q}» → источник {source}"))


def retrieve(s):
    docs = search(s["query"], s["source"], k=5)
    return dict(docs=docs, mode="rag", costs=[record("retrieve", "FAISS+BM25")],
                log=L("retrieve", f"запрос «{s['query']}»: найдено {len(docs)}: "
                      + "; ".join(d["title"][:50] for d in docs[:3])))


def grade(s):
    if not s["docs"]:
        return dict(is_ok=False, log=L("grade", "ничего не найдено"))
    out, c = call("small", "grade",
        "Содержит ли контекст информацию, нужную для ответа на вопрос? Ответь YES или NO.\n\n"
        f"Контекст:\n{ctx(s['docs'], 3, 400)}\n\nВопрос: {s['subs'][s['i']]}")
    ok = "YES" in out.upper()
    return dict(is_ok=ok, costs=[c],
                log=L("grade", "контекст релевантен" if ok else "контекст не подходит"))


def after_grade(s):
    if s.get("is_ok"):
        return "generate"
    return "rewrite" if s["tries"] < 2 else "web"


def rewrite(s):
    out, c = call("small", "rewrite",
        "Перепиши поисковый запрос так, чтобы найти ответ в базе статей: ключевые слова, "
        "синонимы, без лишнего. Для тем про программирование используй английские термины "
        "(python, asyncio, docker). Ответь только запросом"
        + (" на английском" if s["source"] == "science" else "")
        + f".\n\nИсходный вопрос: {s['subs'][s['i']]}\nПрошлый запрос: {s['query']}")
    new_q = out.strip().strip('"«» \n')
    return dict(query=new_q, tries=s["tries"] + 1, costs=[c],
                log=L("rewrite", f"попытка {s['tries'] + 1}: новый запрос «{new_q}»"))


def web(s):
    res = web_search(s["subs"][s["i"]])
    docs = [dict(title=r["title"], url=r["url"], text=r["text"], source="web", search_type="web")
            for r in res]
    mode = "web" if docs else "llm"
    return dict(docs=docs, mode=mode, costs=[record("web_search", "DuckDuckGo")],
                log=L("web", "в базе ответа нет → " + (f"веб-поиск: {len(docs)} результатов"
                                                       if docs else "веб пуст, отвечаю силами LLM")))


def generate(s):
    q = s["subs"][s["i"]]
    strict = " Опирайся строго на контекст, не добавляй фактов от себя." if s["regen"] else ""
    if s["mode"] == "llm":
        prompt = f"Ответь на вопрос по-русски. Укажи, что ответ дан без источников.\n\nВопрос: {q}"
    else:
        prompt = ("Ответь на вопрос по-русски, используя только контекст. Если ответа в "
                  f"контексте нет, так и скажи. Ссылайся на источники как [номер].{strict}\n\n"
                  f"Контекст:\n{ctx(s['docs'], 4, 600)}\n\nВопрос: {q}")
    out, c = call("large", "generate", prompt)
    return dict(answer_i=out, costs=[c], log=L("generate", f"сгенерирован ответ ({s['mode']})"))


def check(s):
    if s["mode"] != "rag":
        return dict(ok_answer=True, no_answer=False,
                    log=L("check", "проверка пропущена (нет источников из базы)"))
    if NOANS.search(s["answer_i"]):
        return dict(ok_answer=False, no_answer=True,
                    log=L("check", "ответ сообщает, что в контексте ничего нет → возврат к поиску"))
    out, c = call("small", "check",
        "Все ли утверждения ответа подтверждены контекстом? Ответь YES или NO.\n\n"
        f"Контекст:\n{ctx(s['docs'], 3, 400)}\n\nОтвет: {s['answer_i']}")
    ok = "YES" in out.upper()
    return dict(ok_answer=ok, no_answer=False, costs=[c],
                log=L("check", "ответ подтверждён контекстом" if ok
                      else "ответ не подтверждён → перегенерация"))


def after_check(s):
    if s.get("no_answer"):
        return "rewrite" if s["tries"] < 2 else "web"
    return "next" if s.get("ok_answer") or s["regen"] >= 1 else "regen"


def regen(s):
    return dict(regen=s["regen"] + 1, log=L("regen", "повторная генерация"))


def nxt(s):
    part = s["partial"] + [dict(q=s["subs"][s["i"]], answer=s["answer_i"], source=s["source"],
                                mode=s["mode"], docs=s["docs"])]
    return dict(partial=part, i=s["i"] + 1, log=L("next", f"подвопрос {s['i'] + 1} готов"))


def more(s):
    return "route" if s["i"] < len(s["subs"]) else "compose"


def compose(s):
    p = s["partial"]
    ans = p[0]["answer"] if len(p) == 1 else "\n\n".join(f"**{x['q']}**\n\n{x['answer']}" for x in p)
    return dict(answer=ans, log=L("compose", "итоговый ответ собран"))


# ---------- граф ----------

g = StateGraph(S)
for name, fn in [("classify", classify), ("chat", chat), ("spellcheck", spellcheck),
                 ("decompose", decompose), ("route", route), ("retrieve", retrieve),
                 ("grade", grade), ("rewrite", rewrite), ("web", web), ("generate", generate),
                 ("check", check), ("regen", regen), ("next", nxt), ("compose", compose)]:
    g.add_node(name, fn)

g.add_edge(START, "classify")
g.add_conditional_edges("classify", lambda s: "spellcheck" if s["is_search"] else "chat")
g.add_edge("chat", END)
g.add_edge("spellcheck", "decompose")
g.add_edge("decompose", "route")
g.add_edge("route", "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "web": "web"})
g.add_edge("rewrite", "retrieve")
g.add_edge("web", "generate")
g.add_edge("generate", "check")
g.add_conditional_edges("check", after_check,
                        {"next": "next", "regen": "regen", "rewrite": "rewrite", "web": "web"})
g.add_edge("regen", "generate")
g.add_conditional_edges("next", more, {"route": "route", "compose": "compose"})
g.add_edge("compose", END)

graph = g.compile()
