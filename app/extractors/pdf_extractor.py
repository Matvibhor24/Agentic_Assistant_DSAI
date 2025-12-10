from typing import Tuple
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO

def extract_pdf_text_from_bytes(data: bytes) -> Tuple[str, float]:
    text_parts = []

    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_parts.append(txt)

    text = "\n".join(text_parts).strip()

    if len(text) < 30:
        images = convert_from_bytes(data)
        ocr_texts = []
        confidences = []
        for img in images:
            t = pytesseract.image_to_string(img)
            ocr_texts.append(t)
            try:
                d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                confs = [int(c) for c in d["conf"] if c != "-1"]
                if confs:
                    confidences.append(sum(confs) / len(confs))
            except Exception:
                pass
        text = "\n".join(ocr_texts).strip()
        conf = sum(confidences) / max(len(confidences), 1) if confidences else 0.0
    else:
        conf = 1.0  # assume good if text layer present

    return text, conf
