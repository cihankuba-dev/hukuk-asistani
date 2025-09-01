import os
import tempfile
import faiss
import pickle
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import openpyxl
from pptx import Presentation

# Google Drive
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS index (hafıza için)
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

# 🔹 Yardımcı Fonksiyon: Dosya Metni Çıkarma
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
        elif ext in ["txt", "rtf", "md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"[HATA] {filename}: {e}")
    return text.strip()

def extract_text(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    content = b"".join(file.file.readlines())
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    text = extract_text_from_path(tmp_path, file.filename)
    os.remove(tmp_path)
    return text

# 🔹 OpenAI Embedding
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 🔹 Ingest endpoint (tek dosya)
@app.post("/ingest")
async def ingest(file: UploadFile):
    global index, metadata
    text = extract_text(file)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for chunk in chunks:
        vector = embed_text(chunk)
        index.add([vector])
        metadata.append({"text": chunk, "file": file.filename})
    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, metadata), f)
    return {"status": "ok", "files_ingested": 1, "chunks_total": len(metadata)}

# 🔹 Ingest Google Drive (klasör ID’den)
@app.post("/ingest_drive")
async def ingest_drive(folder_id: str = Form(...)):
    global index, metadata
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    count = 0

    for file in file_list:
        fname = file['title']
        ext = fname.split(".")[-1].lower()
        if ext not in ["pdf", "docx", "xlsx", "pptx", "txt", "rtf", "md"]:
            continue
        fpath = os.path.join(tempfile.gettempdir(), fname)
        file.GetContentFile(fpath)
        text = extract_text_from_path(fpath, fname)
        os.remove(fpath)

        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            vector = embed_text(chunk)
            index.add([vector])
            metadata.append({"text": chunk, "file": fname})
        count += 1

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, metadata), f)

    return {"status": "ok", "files_ingested": count, "chunks_total": len(metadata)}

# 🔹 Dilekçe hazırlama
@app.post("/petition")
async def petition(prompt: str = Form(...)):
    messages = [
        {"role": "system", "content": "Sen deneyimli bir Türk hukuk asistanısın. Her çıktının sonuna 'Av. Mehmet Cihan KUBA' imzasını ekle."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return {"draft": response.choices[0].message.content}

# 🔹 Belge özetleme
@app.post("/summarize")
async def summarize(file: UploadFile):
    text = extract_text(file)
    messages = [
        {"role": "system", "content": "Sen deneyimli bir hukuk asistanısın. Belgeleri analiz edip özet çıkar."},
        {"role": "user", "content": f"Şu belgeyi özetle: {text[:4000]}"}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return {"summary": response.choices[0].message.content}

# 🔹 Arşivden dilekçe taslağı
@app.post("/draft_from_file")
async def draft_from_file(file: UploadFile):
    text = extract_text(file)
    vector = embed_text(text)
    D, I = index.search([vector], k=5)
    context = "\n".join([metadata[i]["text"] for i in I[0] if i < len(metadata)])
    messages = [
        {"role": "system", "content": "Sen deneyimli bir hukuk asistanısın. Her çıktının sonuna 'Av. Mehmet Cihan KUBA' imzasını ekle."},
        {"role": "user", "content": f"Şu dosya metnine ve geçmiş arşivime göre dilekçe hazırla:\n\n{context}\n\nDosya: {text[:2000]}"}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return {"draft": response.choices[0].message.content}

# 🔹 Mevzuat & içtihat arama
@app.post("/law_search")
async def law_search(query: str = Form(...)):
    messages = [
        {"role": "system", "content": "Sen deneyimli bir hukuk araştırma asistanısın. Güncel mevzuat ve içtihatlardan alıntılarla özet ver."},
        {"role": "user", "content": f"{query} hakkında mevzuat ve içtihat ara."}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return {"result": response.choices[0].message.content}
