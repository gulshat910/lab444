import os
import faiss
from langchain_community.docstore.base import Docstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from src.db import connect, row_to_doc, COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FILE = os.path.join(BASE_DIR, "data", "faiss.index")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})


class SQLiteDocstore(Docstore):
    """Хранилище документов LangChain поверх таблицы metadata."""

    def search(self, search):
        db = connect()
        row = db.execute(f"SELECT {COLS} FROM metadata WHERE id=?",
                         (int(search),)).fetchone()
        db.close()
        return row_to_doc(row, 0, "vector") if row else f"ID {search} not found."


class IdentityMap(dict):
    """В индексе уже лежат id строк SQLite (IndexIDMap), преобразование не нужно."""

    def __getitem__(self, key):
        return int(key)


index = faiss.read_index(INDEX_FILE)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=SQLiteDocstore(),
    index_to_docstore_id=IdentityMap(),
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    normalize_L2=False)

print("Индекс содержит", index.ntotal, "векторов")
