# tez dünyası
import os
import glob
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from transformers import pipeline
from rake_nltk import Rake
import yake
import pytesseract
from pdf2image import convert_from_path
from fastapi import FastAPI, File, UploadFile, Form, Body, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pkg_resources
app = FastAPI(title="Tez Düyası")
from fastapi.staticfiles import StaticFiles
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend") 

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Statik dosya ve template desteği
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# PDF'den metin çıkarma
def extract_text_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception:
        return ""

def extract_text_pdfminer(pdf_path):
    try:
        return pdfminer_extract_text(pdf_path)
    except Exception:
        return ""

def ocr_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang="tur")
        full_text += f"\n--- Sayfa {i+1} ---\n{text}"
    return full_text

def metin_ozetle(text):
    summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
    ozet = summarizer(text, max_length=60, min_length=10, do_sample=False)
    return ozet[0]['summary_text']

def anahtar_kelime_rake(text):
    r = Rake(language='turkish')
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def anahtar_kelime_yake(text, dil="tr"):
    kw_extractor = yake.KeywordExtractor(lan=dil, n=1, top=10)
    keywords = kw_extractor.extract_keywords(text)
    return [kelime for kelime, skor in keywords]

def kategori_bul(text):
    if "makine öğrenmesi" in text.lower():
        return "Yapay Zeka"
    elif "biyoloji" in text.lower():
        return "Biyoloji"
    else:
        return "Diğer"

def kategori_ml(text):
    classifier = pipeline("text-classification", model="dbmdz/bert-base-turkish-cased")
    sonuc = classifier(text[:512])
    return sonuc[0]['label']

def oatd_tez_ara(keyword, max_results=5):
    url = f"https://oatd.org/oatd/search?q={keyword.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tezler = []
    for result in soup.select(".record")[:max_results]:
        title = result.select_one(".title").get_text(strip=True)
        author = result.select_one(".author").get_text(strip=True) if result.select_one(".author") else ""
        tezler.append({"title": title, "author": author})
    return tezler

def core_tez_ara(keyword, max_results=5):
    url = f"https://core.ac.uk/search?q={keyword.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tezler = []
    for result in soup.select(".search-result__title")[:max_results]:
        title = result.get_text(strip=True)
        tezler.append({"title": title, "source": "CORE"})
    return tezler

def tez_konusu_oner(user_keywords, pdf_folder=UPLOAD_DIR):
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")
    konu_onerileri = []
    for pdf_path in pdf_files:
        text = extract_text_pypdf2(pdf_path)
        keywords = set(anahtar_kelime_yake(text))
        ortak = keywords.intersection(set(user_keywords))
        if ortak:
            konu_onerileri.append({
                "pdf": os.path.basename(pdf_path),
                "onerilen_konu": f"{', '.join(ortak)} üzerine yeni bir tez çalışması"
            })
    if not konu_onerileri:
        konu_onerileri.append({
            "pdf": None,
            "onerilen_konu": f"{', '.join(user_keywords)} anahtar kelimeleriyle yeni bir tez konusu araştırabilirsiniz."
        })
    return konu_onerileri

@app.get("/", response_class=HTMLResponse)
async def anasayfa(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Sadece PDF dosyası yükleyebilirsiniz."})
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"message": "Dosya başarıyla yüklendi.", "filename": file.filename}

@app.post("/extract-text/")
async def extract_text(filename: str = Form(...), method: str = Form("pypdf2")):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Dosya bulunamadı."})
    if method == "pypdf2":
        text = extract_text_pypdf2(file_path)
    elif method == "pdfminer":
        text = extract_text_pdfminer(file_path)
    elif method == "ocr":
        text = ocr_pdf(file_path)
    else:
        return JSONResponse(status_code=400, content={"error": "Geçersiz method."})
    return {"text": text}

@app.post("/analyze/")
async def analyze_text(text: str = Form(...)):
    ozet = metin_ozetle(text[:1000])
    rake_keywords = anahtar_kelime_rake(text)
    yake_keywords = anahtar_kelime_yake(text)
    kategori = kategori_bul(text)
    kategori_ml_label = kategori_ml(text)
    return {
        "ozet": ozet,
        "rake_keywords": rake_keywords,
        "yake_keywords": yake_keywords,
        "kategori_kural": kategori,
        "kategori_ml": kategori_ml_label
    }

@app.post("/tez-oner/")
async def tez_oner(keywords: list = Body(..., example=["makine öğrenmesi", "biyomedikal"])):
    return tez_konusu_oner(keywords)

@app.get("/oatd")
async def tez_ara_oatd(q: str = Query(...)):
    return oatd_tez_ara(q)

@app.get("/core")
async def tez_ara_core(q: str = Query(...)):
    return core_tez_ara(q)

@app.get("/health", tags=["Kontrol"])
async def health_check():
    return {"status": "ok"}

@app.get("/requirements", tags=["Kontrol"])
async def list_requirements():
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    return installed_packages

@app.get("/info", tags=["Kontrol"])
async def project_info():
    return {
        "proje": "Tez Analiz ve Öneri API",
        "aciklama": "Bu API, tez PDF'lerinden metin çıkarma, özetleme, anahtar kelime çıkarımı, tez arama ve tez önerisi işlemleri için geliştirilmiştir.",
        "swagger": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "requirements": "/requirements"
    }
