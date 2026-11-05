import ast, json, os, sqlite3, tarfile, warnings
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

D = r"C:\Users\Gulshat\Downloads"
HABR = os.path.join(D, "10_000-постов-Хабр.csv.bz2")
NEWS = os.path.join(D, "10_000-постов-Лента.csv.bz2")
ARXIV = os.path.join(D, "10_000-статей-arxiv2.tar.gz")
LIMIT = 10000


def html_to_text(html):
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    for pre in soup.find_all("pre"):
        pre.replace_with("\n```\n" + pre.get_text() + "\n```\n")
    for t in soup.find_all(["p", "li", "div", "blockquote", "h1", "h2", "h3", "h4", "tr"]):
        t.append("\n\n")
    text = "\n".join(l.rstrip() for l in soup.get_text().splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def split(text, size=1000):
    parts = []
    for p in text.split("\n\n"):
        while len(p) > size:
            parts.append(p[:size]); p = p[size:]
        if p.strip():
            parts.append(p)
    out, buf = [], ""
    for p in parts:
        if buf and len(buf) + len(p) > size:
            out.append(buf.strip()); buf = ""
        buf += p + "\n\n"
    if buf.strip():
        out.append(buf.strip())
    return out


def lst(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except Exception:
        return []


def habr_docs():
    df = pd.read_csv(HABR, compression="bz2")
    for r in df.itertuples():
        yield dict(source="habr", doc_id=str(r.post_id), title=str(r.post_title).replace("\xa0", " "),
                   url=f"https://habr.com/ru/articles/{r.post_id}/", date=str(r.post_publish_date)[:10],
                   topic=", ".join(lst(r.post_hubs)[:3]), text=html_to_text(r.post_html))


def news_docs():
    df = pd.read_csv(NEWS, compression="bz2")
    for i, r in enumerate(df.itertuples()):
        yield dict(source="news", doc_id=str(i), title=str(r.title), url=str(r.url),
                   date=str(r.date), topic=str(r.topic), text=str(r.text))


def science_docs():
    with tarfile.open(ARXIV, "r:gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".json"):
                continue
            meta = json.load(t.extractfile(m))["meta"]
            yield dict(source="science", doc_id=meta.get("236") or m.name,
                       title=" ".join(str(meta.get("202", "")).split()),
                       url=meta.get("url", ""), date=str(meta.get("203", "")),
                       topic=str(meta.get("230", "")),
                       text=" ".join(str(meta.get("81", "")).split()))


os.makedirs("data", exist_ok=True)
db = sqlite3.connect("data/lab4.db")
db.executescript("""
DROP TABLE IF EXISTS fts; DROP TABLE IF EXISTS metadata;
CREATE TABLE metadata(id INTEGER PRIMARY KEY, source TEXT, doc_id TEXT, title TEXT, url TEXT,
    date TEXT, topic TEXT, chunk_id INTEGER, text TEXT);
CREATE INDEX idx_source ON metadata(source);
""")

for gen in (habr_docs, news_docs, science_docs):
    seen, n = set(), 0
    for d in gen():
        if d["doc_id"] in seen or len(d["text"]) < 100:
            continue
        seen.add(d["doc_id"])
        db.executemany(
            "INSERT INTO metadata(source,doc_id,title,url,date,topic,chunk_id,text) VALUES (?,?,?,?,?,?,?,?)",
            [(d["source"], d["doc_id"], d["title"], d["url"], d["date"], d["topic"], i, c)
             for i, c in enumerate(split(d["text"]))])
        n += 1
        if n >= LIMIT:
            break
    db.commit()
    print(d["source"], "документов:", n, flush=True)

db.executescript("""
CREATE VIRTUAL TABLE fts USING fts5(title, text, content='metadata', content_rowid='id', tokenize='unicode61');
INSERT INTO fts(fts) VALUES('rebuild');
""")
db.commit()
print("Фрагментов:", db.execute("SELECT count(*) FROM metadata").fetchone()[0])
