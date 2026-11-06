import os
import faiss, numpy as np, sqlite3
from sentence_transformers import SentenceTransformer

# Модель эмбеддингов: многоязычная, превращает текст в вектор из 384 чисел, работает на CPU
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS = 3      # векторизуем только первые 3 фрагмента каждого документа (экономия времени на CPU);
                # всё остальное находит поиск по словам (BM25)
BATCH = 256     # сколько фрагментов читаем из базы за один раз
IDX = os.path.join("data", "lab4.faiss")   # файл, куда сохраняем векторный индекс

model = SentenceTransformer(MODEL, device="cpu")
db = sqlite3.connect("data/lab4.db")

# Если индекс уже частично построен, загружаем его и продолжаем с последнего id
# (так скрипт можно останавливать и запускать заново)
if os.path.exists(IDX):
    index = faiss.read_index(IDX)
    last = int(faiss.vector_to_array(index.id_map).max()) if index.ntotal else 0
else:
    # IndexFlatIP: точный поиск по скалярному произведению (для нормализованных векторов это косинусная близость)
    # IndexIDMap: позволяет хранить в индексе наши номера (id строк из таблицы metadata)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
    last = 0

# Сколько фрагментов ещё нужно векторизовать
total = db.execute("SELECT count(*) FROM metadata WHERE chunk_id<? AND id>?", (CHUNKS, last)).fetchone()[0]
print("Осталось:", total)

# Берём из базы id, заголовок и текст фрагментов по порядку id
cur = db.execute("SELECT id,title,text FROM metadata WHERE chunk_id<? AND id>? ORDER BY id", (CHUNKS, last))
done = 0
while True:
    rows = cur.fetchmany(BATCH)
    if not rows:
        break
    # К тексту добавляем заголовок, чтобы вектор «знал» тему статьи.
    # normalize_embeddings=True: векторы приводятся к длине 1 (так близость = скалярное произведение)
    v = model.encode([f"{t}\n{x}" for _, t, x in rows], normalize_embeddings=True, batch_size=64)
    # Добавляем векторы в индекс вместе с id из базы: по ним потом достаём текст из SQLite
    index.add_with_ids(np.asarray(v, dtype="float32"), np.array([r[0] for r in rows], dtype="int64"))
    done += len(rows)
    if done % (BATCH * 10) == 0:      # периодически сохраняем индекс на диск
        faiss.write_index(index, IDX); print(f"{done}/{total}", flush=True)

faiss.write_index(index, IDX)
print("Готово. Векторов:", index.ntotal)
