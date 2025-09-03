import os
import io
import tempfile
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PyPDF2 import PdfReader
import httpx
from bs4 import BeautifulSoup
import docx
import openpyxl
from pptx import Presentation

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS index
INDEX_FILE = "faiss_index.pkl"
dimension = 1536
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        index, metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dimension)
    metadata = []

# FastAPI
app = FastAPI(title="⚖️ Hukuk Asistanı")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "⚖️ Hukuk Asistanı aktif ve çalışıyor!"}

# =========================
# Mevzuat ve İçtihat Modülleri (güncel)
# =========================

async def fetch_mevzuat(query: str):
    url = f"https://www.mevzuat.gov.tr/arama?aranan={query}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=60)
            soup = BeautifulSoup(r.text, "html.parser")
            results = [a.text.strip() for a in soup.select("a")]
        return results[:5] if results else ["Sonuç bulunamadı."]
    except Exception as e:
        return [f"Mevzuat bilgisi alınamadı: {e}"]

async def search_ictihat(keyword: str, limit: int = 3) -> list[str]:
    url = f"https://karararama.yargitay.gov.tr/Yargitay-Karar-Forumu?q={keyword}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=100)
            if resp.status_code != 200:
                return [f"Emsal karar bulunamadı ({keyword})."]

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for div in soup.select("div.kararOzet")[:limit]:
                results.append(div.get_text(strip=True))
            return results if results else [f"Emsal karar bulunamadı ({keyword})."]
    except Exception as e:
        return [f"İçtihat bilgisi alınamadı: {e}"]

# =========================
# Dosya Metin Çıkarma
# =========================

def extract_text_from_path(path, filename):
    ext = filename.split(".")[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == "docx":
            doc = docx.Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "xlsx":
            wb = openpyxl.load_workbook(path)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows(values_only=True):
                    text += " ".join([str(cell) for cell in row if cell]) + "\n"
        elif ext == "pptx":
            prs = Presentation(path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        elif ext in ["txt", "rtf", "md", "udf"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"[HATA] {filename}: {e}")
    return text.strip()

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 🔹 Google Drive servis bağlantısı
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)

# =========================
# API Endpoints
# =========================

@app.post("/ingest")
async def ingest(file: UploadFile):
    global index, metadata
    ext = file.filename.split(".")[-1].lower()
    content = b"".join(file.file.readlines())
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = extract_text_from_path(tmp_path, file.filename)
    os.remove(tmp_path)

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for chunk in chunks:
        vector = embed_text(chunk)
        index.add(np.array([vector], dtype="float32"))
        metadata.append({"text": chunk, "file": file.filename})

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, metadata), f)
    return {"status": "ok", "files_ingested": 1, "chunks_total": len(metadata)}

@app.post("/ingest_drive")
async def ingest_drive(folder_id: str = Form(...)):
    global index, metadata
    service = get_drive_service()

    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get("files", [])
    count = 0

    for file in files:
        fname = file["name"]
        ext = fname.split(".")[-1].lower()

        if ext not in ["pdf", "docx", "xlsx", "pptx", "txt", "rtf", "md"]:
            continue

        request = service.files().get_media(fileId=file["id"])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)

        temp_path = os.path.join(tempfile.gettempdir(), fname)
        with open(temp_path, "wb") as f:
            f.write(fh.read())

        text = extract_text_from_path(temp_path, fname)
        os.remove(temp_path)

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            vector = embed_text(chunk)
            index.add(np.array([vector], dtype="float32"))
            metadata.append({"text": chunk, "file": fname})
        count += 1

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, metadata), f)

    return {"status": "ok", "files_ingested": count, "chunks_total": len(metadata)}

@app.post("/petition")
async def petition(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "Sen deneyimli bir Türk hukuk asistanısın. Her çıktının sonuna 'Av. Mehmet Cihan KUBA' imzasını ekle."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model="gpt-5-128k", messages=messages)
    return {"draft": response.choices[0].message.content}

@app.post("/summarize")
async def summarize(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    content = b"".join(file.file.readlines())
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = extract_text_from_path(tmp_path, file.filename)
    os.remove(tmp_path)

    messages = [
        {"role": "system", "content": "Sen deneyimli bir hukuk asistanısın. Belgeleri analiz edip özet çıkar."},
        {"role": "user", "content": f"Şu belgeyi özetle: {text[:4000]}"}
    ]
    response = client.chat.completions.create(model="gpt-5-128k", messages=messages)
    return {"summary": response.choices[0].message.content}

@app.post("/draft_from_file")
async def draft_from_file(file: UploadFile, type: str = Form(...)):
    try:
        # 🔹 Dosya oku
        ext = file.filename.split(".")[-1].lower()
        content = b"".join(file.file.readlines())
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = extract_text_from_path(tmp_path, file.filename)
        os.remove(tmp_path)

        # 🔹 Arşivden içerik çek
        vector = embed_text(text)
        D, I = index.search(np.array([vector], dtype="float32"), k=5)
        context = "\n".join([metadata[i]["text"] for i in I[0] if i < len(metadata)])

        # 🔹 Arşivden üslup örnekleri (max 3)
        style_examples = "\n\n".join([
            m["text"] for m in metadata 
            if m["file"].lower().endswith((".docx", ".pdf"))
        ][:3])

        # 🔹 Mevzuat & İçtihat
        mevzuat_bilgisi = await fetch_mevzuat(type)
        ictihatlar = await search_ictihat(type)

        # 🔹 Prompt
        messages = [
            {"role": "system", "content": "Sen deneyimli bir Türk hukuk asistanısın. "
                                          "Her çıktının sonuna 'Av. Mehmet Cihan KUBA' imzasını ekle."},
            {"role": "user", "content": f"Belge türü: {type}\n\n"
                                        f"Arşivden ilgili içerik:\n{context}\n\n"
                                        f"Mevzuat bilgisi:\n{mevzuat_bilgisi}\n\n"
                                        f"İçtihatlar:\n{ictihatlar}\n\n"
                                        f"Üslup örnekleri (bunlara benzer yaz):\n{style_examples}\n\n"
                                        f"Dosya içeriği:\n{text[:2000]}\n\n"
                                        f"Lütfen yukarıdaki tüm bilgilerden faydalanarak ayrıntılı {type} hazırla."},
        ]

        response = client.chat.completions.create(model="gpt-5-128k", messages=messages)
        return {"draft": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

@app.post("/law_search")
async def law_search(query: str = Form(...)):
    messages = [
        {"role": "system", "content": "Sen deneyimli bir hukuk araştırma asistanısın. Güncel mevzuat ve içtihatlardan alıntılarla özet ver."},
        {"role": "user", "content": f"{query} hakkında mevzuat ve içtihat ara."}
    ]
    response = client.chat.completions.create(model="gpt-5-128k", messages=messages)
    return {"result": response.choices[0].message.content}
