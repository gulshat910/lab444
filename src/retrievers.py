import re
from langchain_core.retrievers import BaseRetriever
from src.db import connect, row_to_doc, COLS
from src.vectorstore import vector_store

STOP = {"что", "как", "какой", "какая", "какие", "каком", "какое", "какую", "это",
        "для", "при", "или", "его", "она", "они", "был", "была", "были", "кто",
        "где", "когда", "сколько", "через", "про", "над", "под", "все", "так"}


def make_match(query):
    """Запрос -> FTS5. Префикс из 6 букв даёт грубую морфологию."""
    words = [w for w in re.findall(r"\w+", query.lower())
             if len(w) >= 3 and w not in STOP]
    terms = list(dict.fromkeys(w[:6] for w in words))
    return " OR ".join(f'"{t}"*' for t in terms)


class FTS5Retriever(BaseRetriever):
    """BM25 по всему корпусу (SQLite FTS5) как LangChain-ретривер."""
    top_k: int = 20

    def _get_relevant_documents(self, query, *, run_manager=None):
        match = make_match(query)
        if not match:
            return []
        cols = ", ".join("m." + c.strip() for c in COLS.split(","))
        db = connect()
        rows = db.execute(
            f"SELECT {cols}, bm25(fts, 3.0, 1.0) AS s "
            "FROM fts JOIN metadata m ON m.id = fts.rowid "
            "WHERE fts MATCH ? ORDER BY s LIMIT ?", (match, self.top_k)).fetchall()
        db.close()
        return [row_to_doc(r, -r[9], "bm25") for r in rows]


vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
bm25_retriever = FTS5Retriever(top_k=20)
