import json, os, sqlite3
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(BASE_DIR, "data", "metadata.db")

COLS = "id, article_id, title, url, published, hubs, tags, chunk_id, text"


def connect():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def row_to_doc(row, score, search_type):
    return Document(page_content=row[8], metadata={
        "id": row[0], "article_id": row[1], "title": row[2], "url": row[3],
        "published": row[4], "hubs": json.loads(row[5]), "tags": json.loads(row[6]),
        "chunk_id": row[7], "score": float(score), "search_type": search_type})
