# main.py   ← rename the file to main.py if it's currently named app.py or something else
# Legal-BERT + PDF Contract Analysis FastAPI backend

import sys
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining
import numpy as np
import fitz  # PyMuPDF

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MY-LogiC Contract Intelligence API",
    description="PDF Contract Upload + Analysis using LEGAL-BERT",
    version="1.0",
    debug=True
)

# ── CORS (allows frontend on different port/origin) ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only — tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load LEGAL-BERT (optional for now — can be used later for better analysis) ──
device = torch.device("cpu")
model = None
tokenizer = None

try:
    logger.info("Loading LEGAL-BERT (may take 30–90s first time)...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", use_fast=True)
    model = AutoModelForPreTraining.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model.to(device)
    model.eval()
    logger.info(f"LEGAL-BERT loaded on {device}")
except Exception as e:
    logger.warning("LEGAL-BERT failed to load — continuing without it", exc_info=True)

# ── Simple mock clause types & risk keywords (expand this later) ──
CLAUSE_PATTERNS = {
    "Limitation of Liability": ["limit", "liability", "damages", "consequential", "indirect"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless"],
    "Termination": ["terminate", "termination", "cancel", "end this agreement"],
    "Governing Law": ["governed by", "governing law", "jurisdiction"],
    "Confidentiality": ["confidential", "nda", "non-disclosure"],
    "Payment Terms": ["payment", "fee", "invoice", "due date"],
}

def simple_risk_analysis(text: str) -> tuple[str, int, str]:
    """Very basic heuristic — replace with REAL ML later"""
    text_lower = text.lower()
    
    best_type = "General Clause"
    best_score = 0
    for clause_type, keywords in CLAUSE_PATTERNS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > best_score:
            best_score = matches
            best_type = clause_type
    
    confidence = min(40 + best_score * 15, 95)
    risk_reason = "High risk" if "liability" in text_lower or "indemnify" in text_lower else "Medium risk – review recommended"
    
    return best_type, confidence, risk_reason

# ── PDF Text Extraction with PyMuPDF ─────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n\n"
        doc.close()
        return full_text.strip()
    except Exception as e:
        logger.error("PDF extraction failed", exc_info=True)
        raise HTTPException(400, f"Failed to read PDF: {str(e)}")

# ── Upload & Analyze Endpoint ────────────────────────────────────
@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed")

    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 10MB)")

        extracted_text = extract_text_from_pdf(contents)

        paragraphs = [p.strip() for p in extracted_text.split("\n\n") if p.strip() and len(p.strip()) > 30]

        clauses = []
        for para in paragraphs[:15]:
            clause_type, conf, reason = simple_risk_analysis(para)
            clauses.append({
                "clause_type": clause_type,
                "extracted_text": para[:400] + ("..." if len(para) > 400 else ""),
                "confidence_score": conf,
                "risk_flag_reason": reason
            })

        return {"clauses": clauses}

    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        raise HTTPException(500, f"Analysis error: {str(e)}")

# ── Health check ─────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "MY-LogiC Contract Intelligence API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",           # ← FIXED: changed from "app:app" to "main:app"
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )