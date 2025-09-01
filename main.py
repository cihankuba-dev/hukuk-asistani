import os
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import docx
from pdfminer.high_level import extract_text

# OpenAI API Key yükle
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Hukuk Asistanı",
    description="📂 Dosya yükle → ⚖️ Dilekçe taslağı al",
    version="1.0"
)

# iPad / Web erişimi için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dosya okuma fonksiyonu
def read_file(file: UploadFile):
    if file.filename.endswith(".pdf"):
        text = extract_text(file.file)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.file.read().decode("utf-8")
    return text[:8000]  # 8K token sınırı için kesinti

# Tek endpoint: Dosyadan dilekçe hazırla
@app.post("/draft_from_file")
async def draft_from_file(file: UploadFile):
    # 1. Dosya oku
    text = read_file(file)

    # 2. Özet çıkar
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen Türkiye odaklı bir hukuk asistanısın. Belgeleri kısa ve tarafsız özetle."},
            {"role": "user", "content": text}
        ]
    ).choices[0].message.content

    # 3. Dilekçe hazırla
    draft = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen profesyonel bir Türk hukuk dilekçe asistanısın."},
            {"role": "user", "content": f"""
Belge özeti:
{summary}

Arşivden ilgili belgeler:
📂 [Emsal dilekçeler daha sonra eklenecek]

Mevzuat ve içtihat:
⚖️ [Kanun maddeleri]
📑 [İçtihat özetleri]

Görev: Yukarıdaki bilgilerle resmi formatta bir dilekçe taslağı hazırla.
Format: Başlık, taraf bilgileri, olay özeti, hukuki dayanaklar, talepler.
"""}
        ]
    ).choices[0].message.content

    return {
        "summary": summary,
        "draft": draft
    }
