import pandas as pd
import streamlit as st
from src.graph import graph   # скомпилированный граф LangGraph из src/graph.py

# Настройки страницы: заголовок вкладки браузера, значок, широкий режим
st.set_page_config(page_title="RAG на LangGraph", page_icon="🔎", layout="wide")

# дизайн. Задаёт фон, шапку, карточки шагов,
# метки источников, рамки метрик и цвет главной кнопки
st.markdown("""<style>
.stApp {background: linear-gradient(180deg,#f6fdf8 0%,#e8f7ee 100%);}
[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #d5eedd;}
.block-container {padding-top:1.5rem; max-width:1100px;}
.hero {background:linear-gradient(135deg,#16a34a,#059669); color:#fff; padding:1.4rem 1.8rem;
       border-radius:18px; margin-bottom:1rem; box-shadow:0 8px 24px rgba(22,163,74,.25);}
.hero h1 {color:#fff; margin:0; font-size:1.9rem;}
.hero p {color:#dcfce7; margin:.3rem 0 0;}
.step {background:#fff; border:1px solid #d5eedd; border-left:5px solid #22c55e;
       padding:.5rem .9rem; border-radius:10px; margin-bottom:.4rem;
       box-shadow:0 1px 3px rgba(0,0,0,.04);}
.step b {color:#15803d; text-transform:uppercase; font-size:.75rem; letter-spacing:.04em;}
.pill {display:inline-block; padding:1px 10px; border-radius:20px; font-size:.75rem;
       background:#dcfce7; color:#166534; margin-right:6px;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #d5eedd; border-radius:14px;
       padding:.8rem 1rem; box-shadow:0 2px 8px rgba(22,163,74,.10);}
.stButton button {border-radius:10px;}
.stButton button[kind="primary"] {background:#16a34a; border-color:#16a34a;}
.stButton button[kind="primary"]:hover {background:#15803d; border-color:#15803d;}
</style>""", unsafe_allow_html=True)

# Шапка страницы
st.markdown("<div class='hero'><h1>🔎 RAG на LangGraph</h1>"
            "<p>Хабр · Научные статьи arXiv · Новости Lenta.ru</p></div>", unsafe_allow_html=True)

# Контрольный набор вопросов (предлоаем его сами):
# разговорные, по каждому источнику, с опечатками, составной, вопросы вне базы
QUESTIONS = [
    ("Разговорный", "Привет! Что ты умеешь?"),
    ("Разговорный", "Спасибо, понятно!"),
    ("Хабр", "Что пишут про Илона Маска и его состояние?"),
    ("Хабр", "Как работает асинхронность в питоне?"),
    ("Новости", "Что известно про суд Оззи Осборна против таблоида?"),
    ("Наука", "Что известно про адиабатические квантовые вычисления?"),
    ("Опечатки", "Как работаит асинхроность в питоне?"),
    ("Составной", "Что пишут про Илона Маска и что случилось с женой Оззи Осборна?"),
    ("Нет в базе", "Какая столица Австралии?"),
    ("Нет в базе", "В какой компании работал Сридхар Рамасвами?"),
]

# Streamlit перерисовывает страницу при каждом действии, поэтому историю запросов
# храним в session_state: из неё строятся вкладки «Ход выполнения» и Cost Dashboard
if "runs" not in st.session_state:
    st.session_state.runs = []


def run_question(question, show=None):
    """Выполняет граф, возвращает итоговое состояние. show(e) выводит шаг на экран."""
    final = {"log": [], "costs": []}
    # stream_mode="updates": граф отдаёт обновление после КАЖДОГО узла,
    # поэтому шаги появляются на экране по мере выполнения (пункт 8)
    # recursion_limit защищает от бесконечных циклов в графе
    for upd in graph.stream({"question": question, "i": 0, "partial": [], "log": [], "costs": []},
                            stream_mode="updates", config={"recursion_limit": 80}):
        for node, data in upd.items():
            for k, v in data.items():
                if k in ("log", "costs"):
                    final[k] += v        # журнал и затраты накапливаются
                else:
                    final[k] = v         # остальные поля перезаписываются
            for e in data.get("log", []):
                if show:
                    show(e)
    st.session_state.runs.append(dict(question=question, **final))   # сохраняем в историю
    return final


def step_html(e):
    """Оформление одного шага графа в виде карточки."""
    return f"<div class='step'><b>{e['node']}</b><br>{e['text']}</div>"


# Боковая панель: контрольные вопросы одной кнопкой, прогон всего набора, очистка истории
with st.sidebar:
    st.header("Контрольный набор")
    for cat, q in QUESTIONS:
        if st.button(f"{cat}: {q}", use_container_width=True, key=q):
            st.session_state.pending = q     # вопрос обработается во вкладке «Чат»
    st.divider()
    run_all = st.button("▶ Прогнать весь набор", type="primary", use_container_width=True)
    if st.button("Очистить историю", use_container_width=True):
        st.session_state.runs = []
        st.rerun()

tab_chat, tab_steps, tab_cost = st.tabs(["💬 Чат", "🧩 Ход выполнения", "💰 Cost Dashboard"])

# ---------- Вкладка «Чат» ----------
with tab_chat:
    # вопрос из поля ввода или из боковой панели
    question = st.chat_input("Задайте вопрос") or st.session_state.pop("pending", None)
    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            # шаги графа выводятся внутри блока статуса по мере выполнения (пункт 8)
            with st.status("Выполняю шаги графа...", expanded=True) as status:
                final = run_question(question, lambda e: st.markdown(step_html(e), unsafe_allow_html=True))
                status.update(label="Готово", state="complete", expanded=False)
            st.markdown(final.get("answer", "—"))       # итоговый ответ
            # источники по каждому подвопросу: откуда взяты (habr/news/science/web) и режим
            for p in final.get("partial", []):
                with st.expander(f"Источники: {p['q']}"):
                    st.markdown(f"<span class='pill'>{p['source']}</span><span class='pill'>{p['mode']}</span>",
                                unsafe_allow_html=True)
                    for i, d in enumerate(p["docs"], 1):
                        st.markdown(f"**[{i}] {d['title']}**  \n{d.get('url', '')}")
                        st.caption(d["text"][:300] + "…")
    elif run_all:
        # прогон всего контрольного набора подряд с индикатором прогресса
        bar = st.progress(0.0)
        for n, (cat, q) in enumerate(QUESTIONS, 1):
            with st.expander(f"{cat}: {q}", expanded=False):
                f = run_question(q, lambda e: st.markdown(step_html(e), unsafe_allow_html=True))
                st.markdown(f.get("answer", "—"))
            bar.progress(n / len(QUESTIONS))
        st.success("Набор выполнен. Смотрите вкладки «Ход выполнения» и «Cost Dashboard».")

# ---------- Вкладка «Ход выполнения» (пункт 8) ----------
with tab_steps:
    if not st.session_state.runs:
        st.info("Задайте вопрос на вкладке «Чат».")
    # журнал шагов каждого запроса, последние сверху
    for n, r in enumerate(reversed(st.session_state.runs), 1):
        with st.expander(r["question"], expanded=(n == 1)):
            for e in r["log"]:
                st.markdown(step_html(e), unsafe_allow_html=True)

# ---------- Вкладка «Cost Dashboard» (пункт 9) ----------
with tab_cost:
    runs = st.session_state.runs
    if not runs:
        st.info("Данных пока нет.")
    else:
        # одна строка таблицы = одна запись о затратах одного шага одного запроса
        rows = [dict(запрос=i + 1, вопрос=r["question"][:40], **c)
                for i, r in enumerate(runs) for c in r["costs"]]
        df = pd.DataFrame(rows)
        df["tokens"] = df.tokens_in + df.tokens_out
        # общие метрики по всей серии запросов
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Запросов", len(runs))
        c2.metric("Токенов", int(df.tokens.sum()))
        c3.metric("Время, с", f"{df.sec.sum():.1f}")
        c4.metric("Стоимость, $", f"{df.cost.sum():.6f}")

        # затраты на каждый этап RAG для последнего запроса
        st.subheader("Последний запрос по этапам")
        last = df[df["запрос"] == len(runs)]
        st.dataframe(last.groupby("node")[["tokens_in", "tokens_out", "sec", "cost"]].sum(),
                     use_container_width=True)
        # затраты на запрос в целом и на серию запросов
        st.subheader("Серия запросов")
        per = df.groupby("запрос")[["cost", "sec", "tokens"]].sum()
        a, b = st.columns(2)
        a.caption("Стоимость по запросам"); a.bar_chart(per["cost"])
        b.caption("Время по запросам, с"); b.bar_chart(per["sec"])
        # какие этапы самые дорогие по всей серии
        st.subheader("По этапам RAG (вся серия)")
        st.dataframe(df.groupby("node")[["tokens", "sec", "cost"]].sum().sort_values("cost", ascending=False),
                     use_container_width=True)
        # сравнение моделей разного размера
        st.subheader("По моделям")
        st.dataframe(df.groupby("model")[["tokens_in", "tokens_out", "sec", "cost"]].sum(),
                     use_container_width=True)
        st.caption("Для локальных моделей цены условные ($ за 1M токенов), задаются в src/llm.py.")
