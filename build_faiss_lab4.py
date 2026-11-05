import os
import faiss, numpy as np, sqlite3
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS = 3
BATCH = 256
IDX = os.path.join("data", "lab4.faiss")

model = SentenceTransformer(MODEL, device="cpu")
db = sqlite3.connect("data/lab4.db")

if os.path.exists(IDX):
    index = faiss.read_index(IDX)
    last = int(faiss.vector_to_array(index.id_map).max()) if index.ntotal else 0
else:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
    last = 0

total = db.execute("SELECT count(*) FROM metadata WHERE chunk_id<? AND id>?", (CHUNKS, last)).fetchone()[0]
print("Осталось:", total)
cur = db.execute("SELECT id,title,text FROM metadata WHERE chunk_id<? AND id>? ORDER BY id", (CHUNKS, last))
done = 0
while True:
    rows = cur.fetchmany(BATCH)
    if not rows:
        break
    v = model.encode([f"{t}\n{x}" for _, t, x in rows], normalize_embeddings=True, batch_size=64)
    index.add_with_ids(np.asarray(v, dtype="float32"), np.array([r[0] for r in rows], dtype="int64"))
    done += len(rows)
    if done % (BATCH * 10) == 0:
        faiss.write_index(index, IDX); print(f"{done}/{total}", flush=True)
faiss.write_index(index, IDX)
print("Готово. Векторов:", index.ntotal)
