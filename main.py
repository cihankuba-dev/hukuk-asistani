import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import docx
from pdfminer.high_level import extract_text

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file(file: UploadFile):
    if file.filename.endswith(".pdf"):
        text = extract_text(file.file)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.file.read().decode("utf-8")
    return text[:8000]

@app.post("/summarize")
async def summarize(file: UploadFile):
    text = read_file(file)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen Türkiye odaklı bir hukuk asistanısın. Kaynaklı ve kısa özet ver."},
            {"role": "user", "content": f"Metni özetle: {text}"}
        ]
    )
    return {"summary": response.choices[0].message.content}

@app.post("/petition")
async def petition(prompt: str = Form(...)):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen profesyonel bir Türk hukuk dilekçe asistanısın."},
            {"role": "user", "content": f"{prompt} için bir dilekçe taslağı hazırla."}
        ]
    )
    return {"draft": response.choices[0].message.content}
