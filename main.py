import os
import tempfile
import zipfile
from lxml import etree
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
from PyPDF2 import PdfReader
import docx
import openpyxl
from pptx import Presentation
import faiss

app = FastAPI(title="⚖️ Hukuk Asistanı")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# Yardımcı Fonksiyonlar
# ==========================

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_word(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_excel(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path)
    text = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.iter_rows(values_only=True):
            text.append(" ".join([str(cell) for cell in row if cell]))
    return "\n".join(text)

def extract_text_from_pptx(file_path: str) -> str:
    prs = Presentation(file_path)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def extract_text_from_udf(file_path: str) -> str:
    """UYAP Doküman Editörü (.udf) dosyalarından metin çıkarır."""
    text_content = []
    with zipfile.ZipFile(file_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                xml_data = z.read(name)
                try:
                    tree = etree.fromstring(xml_data)
                    for elem in tree.iter():
                        if elem.text:
                            text_content.append(elem.text.strip())
                except Exception:
                    continue
    return "\n".join(text_content)

# ==========================
# API Endpoint'leri
# ==========================

@app.post("/summarize")
async def summarize(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Dosya tipine göre metin çıkarma
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(tmp_path)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_word(tmp_path)
    elif file.filename.endswith(".xlsx"):
        text = extract_text_from_excel(tmp_path)
    elif file.filename.endswith(".pptx"):
        text = extract_text_from_pptx(tmp_path)
    elif file.filename.endswith(".udf"):
        text = extract_text_from_udf(tmp_path)
    else:
        return JSONResponse({"error": "Desteklenmeyen dosya türü"})

    # OpenAI ile özet
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen deneyimli bir hukuk asistanısın."},
            {"role": "user", "content": f"Şu belgeyi özetle:\n\n{text[:5000]}"},
        ]
    )

    return {"summary": completion.choices[0].message.content}

@app.post("/petition")
async def petition(prompt: str = Form(...)):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen deneyimli bir hukuk asistanısın. Her dilekçenin sonuna 'Av. Mehmet Cihan KUBA' imzasını ekle."},
            {"role": "user", "content": prompt},
        ]
    )
    return {"draft": completion.choices[0].message.content}
