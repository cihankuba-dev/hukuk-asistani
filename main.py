import os, io, json, uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import docx
from pdfminer.high_level import extract_text
from pptx import Presentation

# ---- Config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCS_PATH  = os.path.join(DATA_DIR, "docs.json")

# ---- Simple FAISS store (on-disk)
try:
    import faiss                   # faiss-cpu
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def load_docs():
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"metas": [], "chunks": []}

def save_docs(store):
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)

def chunk_text(txt, max_chars=1200, overlap=150):
    txt = txt.strip()
    chunks = []
    start = 0
    while start < len(txt):
        end = min(len(txt), start + max_chars)
        chunk = txt[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
        if end == len(txt): break
    return chunks if chunks else [txt]

def embed_texts(texts: List[str]) -> List[List[float]]:
    # ucuz ve iyi: text-embedding-3-small
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in resp.data]

def ensure_faiss(dim):
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS yüklü değil. requirements.txt içinde 'faiss-cpu' var mı?")
    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)
    else:
        idx = faiss.IndexFlatIP(dim)  # cosine için normalize edeceğiz
    return idx

def normalize(vecs):
    import numpy as np
    arr = np.array(vecs).astype("float32")
    norms = (arr**2).sum(axis=1) ** 0.5
    norms[norms == 0] = 1e-10
    arr = arr / norms[:, None]
    return arr

def add_to_index(embeds):
    import numpy as np
    vecs = normalize(embeds)
    idx = ensure_faiss(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, INDEX_PATH)

def search_index(query, k=5):
    import numpy as np
    if not os.path.exists(INDEX_PATH):
        return []
    store = load_docs()
    qv = embed_texts([query])[0]
    qv = normalize([qv])
    idx = faiss.read_index(INDEX_PATH)
    D, I = idx.search(qv, k)
    I = I[0].tolist()
    results = []
    for rank, pos in enumerate(I):
        if pos < len(store["chunks"]):
            results.append({
                "rank": rank+1,
                "score": float(D[0][rank]),
                "text": store["chunks"][pos]["text"],
                "meta": store["chunks"][pos]["meta"]
            })
    return results

# ---- File readers
def read_docx(filelike) -> str:
    # python-docx doğrudan stream ile uğraşır, gerekirse BytesIO
    bio = io.BytesIO(filelike.read())
    doc = docx.Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(filelike) -> str:
    return extract_text(filelike)

def read_pptx(filelike) -> str:
    bio = io.BytesIO(filelike.read())
    prs = Presentation(bio)
    texts = []
    for slide in prs.slides:
        for shp in slide.shapes:
            if hasattr(shp, "text"):
                texts.append(shp.text)
    return "\n".join(texts)

def read_any(upload: UploadFile) -> str:
    name = upload.filename.lower()
    raw = upload.file
    if name.endswith(".pdf"):
        return read_pdf(raw)
    if name.endswith(".docx"):
        return read_docx(raw)
    if name.endswith(".pptx") or name.endswith(".ppt"):
        return read_pptx(raw)
    # txt/rtf/md gibi
    return raw.read().decode("utf-8", errors="ignore")

app = FastAPI(
    title="Hukuk Asistanı (Kişisel – RAG + Üslup)",
    description="iCloud arşivini ingest eder, üslubunu öğrenir, RAG ile dilekçe üretir.",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 1) iCloud klasöründeki dosyaları ingest (öğrenme havuzu)
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    store = load_docs()
    added = 0
    metas = []
    for f in files:
        try:
            text = read_any(f)
            chunks = chunk_text(text)
            meta = {
                "id": str(uuid.uuid4()),
                "filename": f.filename
            }
            for ch in chunks:
                store["chunks"].append({"text": ch, "meta": meta})
            store["metas"].append(meta)
            metas.append(meta)
            added += len(chunks)
        except Exception as e:
            metas.append({"filename": f.filename, "error": str(e)})
    save_docs(store)

    # embedding + index
    embeds = embed_texts([c["text"] for c in store["chunks"]])
    add_to_index(embeds)

    return {
        "status": "ok",
        "files_ingested": len(files),
        "chunks_total": len(store["chunks"]),
        "last_added_chunks": added
    }

# 2) Hızlı özet
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    text = read_any(file)
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Türk hukuk asistanısın. Belgeyi kısa ve tarafsız özetle."},
            {"role": "user", "content": text[:16000]}
        ]
    ).choices[0].message.content
    return {"summary": summary}

# 3) Serbest dilekçe
@app.post("/petition")
async def petition(prompt: str = Form(...)):
    # Üslup için arşivden 3 örnek paragraf çekiyoruz
    examples = search_index("dilekçe üslubumdan örnek", k=3)
    style_snips = "\n\n".join(r["text"] for r in examples) if examples else ""
    draft = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Profesyonel bir Türk hukuk dilekçe asistanısın."},
            {"role": "user", "content": f"Üslup örnekleri (benim arşivimden):\n{style_snips}\n\nGörev: {prompt}\nÜslubumu koru, resmi format, başlıklar, dayanaklar, talepler."}
        ]
    ).choices[0].message.content
    return {"draft": draft}

# 4) Sorgu + RAG arama (arşiv)
@app.get("/search")
async def rag_search(q: str = Query(..., description="Arşivde ara")):
    results = search_index(q, k=5)
    return {"results": results}

# 5) Belgeden Dilekçe (RAG + Üslup)
@app.post("/draft_from_file")
async def draft_from_file(file: UploadFile = File(...)):
    text = read_any(file)
    # Önce özet
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Türk hukuk asistanısın. Belgeleri kısa özetle."},
            {"role": "user", "content": text[:16000]}
        ]
    ).choices[0].message.content

    # Arşivden 5 ilgili parça (emsal/üslup)
    hits = search_index(summary, k=5)
    context = "\n\n".join([f"[{h['meta'].get('filename','arşiv')}] {h['text']}" for h in hits])

    # Güncel mevzuat/karar (şimdilik model tabanlı özet – doğrulama iste)
    law_hint = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Türk hukuk danışmanısın. Mevzuat ve içtihat önerileri ver; metinleri kısa, nokta atışı çıkar. Tarih belirt ve kullanıcıya doğrulama uyarısı yap."},
            {"role":"user","content": f"Şu özet için uygun mevzuat ve içtihat öner: {summary}"}
        ]
    ).choices[0].message.content

    # Dilekçe taslağı
    draft = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"Profesyonel bir Türk hukuk dilekçe asistanısın."},
            {"role":"user","content": f"""
Belge özeti:
{summary}

Arşivden (üslup+emsal) ilgili parçalar:
{context}

Mevzuat & içtihat önerileri (kontrol edilmelidir):
{law_hint}

Görev: Yukarıdaki bilgilerle, benim arşiv üslubuma yakın, resmi formatta kapsamlı bir dilekçe hazırla.
Bölümler: Başlık, taraf bilgileri, olay özeti, hukuki nitelendirme, mevzuat dayanakları, içtihat alıntıları (kısa), sonuç ve talepler, ekler.
"""}
        ]
    ).choices[0].message.content

    return {"summary": summary, "context_used": hits, "law_suggestions": law_hint, "draft": draft}

# 6) Basit mevzuat & içtihat arama (model tabanlı, uyarı içerir)
@app.get("/law_search")
async def law_search(query: str = Query(...)):
    text = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Türk hukuk danışmanısın. Kısa mevzuat maddeleri ve Yargıtay/AYM/Danıştay içtihat özetleri öner. Güncelliği kullanıcı doğrulamalı uyarısı ekle."},
            {"role":"user","content": f"Sorgu: {query}\nÇıktı: Madde metni özeti + 2-3 içtihat özeti + kısa dipnot/uyarı."}
        ]
    ).choices[0].message.content
    return {"results": text}
