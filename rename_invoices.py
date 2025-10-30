import sys, re, os, argparse, csv
import pytesseract
from unidecode import unidecode
import fitz  # PyMuPDF
import cv2
import numpy as np
from dateutil import parser as dparser
from datetime import datetime

DATE_HINT_WORDS = [
    r"datum\s*vystav(e|\u011b)ní", r"vystaveno", r"date\s*of\s*issue",
    r"issue\s*date", r"datum", r"date"
]
INVOICE_HINT_WORDS = [
    r"(\u010díslo|cislo)\s*faktury", r"faktura\s*\u010d\.?", r"faktura\s*#?",
    r"invoice\s*(no\.|number|#)?", r"inv\s*(no\.|number|#)?"
]
SUPPLIER_HINT_WORDS = [r"dodavatel", r"supplier", r"vystavitel"]

COMPANY_MARKERS = [r"s\.r\.o\.", r"a\.s\.", r"sro", r"as", r"se", r"gmbh", r"kg", r"spol\.", r"inc\.", r"ltd\.?"]

DATE_REGEXES = [
    r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b",     # 12.09.2025 / 12-9-25
    r"\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b",       # 2025-09-12
    r"\b(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{2,4})\b" # 12 . 9 . 2025
]

INVOICE_NO_REGEXES = [
    r"(?:\u010díslo|cislo)\s*faktury[:\s]*([A-Z0-9/\-]+)",
    r"faktura\s*(?:\u010d\.|#)?[:\s]*([A-Z0-9/\-]+)",
    r"invoice\s*(?:no\.|number|#)?[:\s]*([A-Z0-9/\-]+)",
    r"\b(?:inv|var|vs)\s*[:#]?\s*([A-Z0-9/\-]{3,})"
]

def normalize_date(any_date_text):
    # Try fast regex parse; fallback to dateutil
    s = any_date_text.strip()
    # Unified: detect dd/mm/yyyy etc.
    for rgx in DATE_REGEXES:
        m = re.search(rgx, s, flags=re.IGNORECASE)
        if m:
            g = m.groups()
            # Heuristics for order
            if len(g) == 3:
                a,b,c = g
                a,b,c = [int(x) for x in (a,b,c)]
                if len(str(g[0]))==4:  # yyyy-mm-dd
                    y,mn,d = a,b,c
                elif len(str(g[2]))==4: # dd-mm-yyyy
                    y,mn,d = c,b,a
                else:  # dd-mm-yy
                    y = 2000 + c if c < 100 else c
                    y,mn,d = y,b,a
                try:
                    return datetime(y,mn,d).strftime("%Y-%m-%d")
                except:
                    pass
    # Fallback
    try:
        dt = dparser.parse(s, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except:
        return None

def render_first_page_as_image(pdf_path, dpi=250):
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        return None
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize & threshold for better OCR
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return thr

def ocr_image(img, tesseract_path=None, lang_hint="ces+eng"):
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    try:
        txt = pytesseract.image_to_string(img, lang=lang_hint)
        return txt
    except Exception as e:
        return ""

def find_first_by_hints(text, hint_words):
    # Return the line following a hint if that helps
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)
    for hw in hint_words:
        m = re.search(hw, joined, flags=re.IGNORECASE)
        if m:
            # Try the next line after hint location
            # Find which line index contains the match
            idx = 0
            pos = m.start()
            acc = 0
            for i,l in enumerate(lines):
                acc += len(l)+1
                if acc > pos:
                    idx = i
                    break
            # Candidate lines near hint
            for j in range(idx, min(idx+4, len(lines))):
                if j!=idx and len(lines[j])>=3:
                    yield lines[j]
    # Fallback: nothing
    return

def extract_date(text):
    # Try hint-based first
    for line in find_first_by_hints(text, DATE_HINT_WORDS):
        dt = normalize_date(line)
        if dt: return dt
    # Global search
    for rgx in DATE_REGEXES:
        for m in re.finditer(rgx, text, flags=re.IGNORECASE):
            dt = normalize_date(m.group(0))
            if dt: return dt
    # Another fallback: whole text parse
    return normalize_date(text)

def extract_invoice_no(text):
    for rgx in INVOICE_NO_REGEXES:
        m = re.search(rgx, text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"[^A-Z0-9/\-]", "", raw.upper())
            if len(raw) >= 3:
                return raw
    # Loose fallback: look for patterns like 2025/00123, FA-2025-001, etc.
    m = re.search(r"\b([A-Z]{1,4}[-/]?\d{2,4}[-/]\d{2,6}|[0-9]{3,}/[0-9]{2,4}|[A-Z0-9]{5,})\b",
                  text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

def clean_company_name(s):
    s = s.strip()
    s = re.sub(r"^[^A-Za-z0-9ÁÉÍÓÚÝČĎĚŇŘŠŤŮŽáéíóúýčďěňřšťůž]+", "", s)
    # Strip emails / phones
    s = re.sub(r"\S+@\S+", "", s)
    s = re.sub(r"\+?\d[\d\s\-]{6,}", "", s)
    # Remove address-like junk
    s = re.sub(r"\b(ulice|street|ul\.|\u010d\.p\.|psc|zip|budova|building)\b.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" .,-")
    # Limit length
    if len(s) > 60: s = s[:60].rstrip()
    return s

def looks_like_company(line):
    if any(re.search(m, line, flags=re.IGNORECASE) for m in COMPANY_MARKERS):
        return True
    # Heuristic: many letters + at least 2 words
    if len(re.findall(r"[A-Za-zÁÉÍÓÚÝČĎĚŇŘŠŤŮŽáéíóúýčďěňřšťůž]", line)) > 8 and len(line.split()) >= 2:
        return True
    return False

def extract_supplier(text):
    # Hint-based
    for cand in find_first_by_hints(text, SUPPLIER_HINT_WORDS):
        cand = clean_company_name(cand)
        if len(cand) >= 3:
            return cand
    # Top region heuristic: scan first ~12 lines for company-like names
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:12]:
        if looks_like_company(l):
            cand = clean_company_name(l)
            if len(cand) >= 3:
                return cand
    # Fallback: try domain name as supplier
    m = re.search(r"\b([a-z0-9\-]+)\.(cz|sk|com|eu|de)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None

def sanitize_filename(s):
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = re.sub(r'[:*?"<>|]', "-", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def build_new_name(date_str, supplier, invno):
    parts = []
    parts.append(date_str or "0000-00-00")
    parts.append(supplier or "Neznamy-dodavatel")
    parts.append(invno or "Bez-cisla")
    name = " ".join(parts) + ".pdf"
    # Unidecode to avoid diacritics if desired (switch off if chces háčky)
    name = unidecode(name)
    return sanitize_filename(name)

def process_pdf(path, tesseract_path=None):
    img = render_first_page_as_image(path)
    if img is None:
        return None, None, None, "NO_PAGE"
    pre = preprocess(img)
    text = ocr_image(pre, tesseract_path=tesseract_path, lang_hint="ces+eng")
    if not text or len(text.strip()) == 0:
        # try without preprocess
        text = ocr_image(img, tesseract_path=tesseract_path, lang_hint="ces+eng")
    # Extract fields
    date = extract_date(text) or ""
    inv = extract_invoice_no(text) or ""
    sup = extract_supplier(text) or ""
    return date, sup, inv, "OK"

def main():
    ap = argparse.ArgumentParser(description="Hromadné přejmenování faktur z PDF pomocí OCR.")
    ap.add_argument("input_dir", help="Složka se vstupními PDF.")
    ap.add_argument("--inplace", action="store_true", help="Skutečně přejmenovat soubory (jinak jen log).")
    ap.add_argument("--tesseract", default=None, help="Cesta k tesseract.exe (Windows).")
    args = ap.parse_args()

    in_dir = args.input_dir
    log_path = os.path.join(in_dir, "_rename_log.csv")
    rows = []
    pdfs = [f for f in os.listdir(in_dir) if f.lower().endswith(".pdf")]

    for f in pdfs:
        full = os.path.join(in_dir, f)
        try:
            date, sup, inv, status = process_pdf(full, tesseract_path=args.tesseract)
            new_name = build_new_name(date, sup, inv) if status == "OK" else ""
            rows.append([f, date, sup, inv, new_name, status])
        except Exception as e:
            rows.append([f, "", "", "", "", f"ERROR:{e}"])

    # Write CSV log
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["original_file", "date", "supplier", "invoice_no", "proposed_name", "status"])
        w.writerows(rows)

    # Do renames if requested
    if args.inplace:
        for orig, date, sup, inv, proposed, status in rows:
            if status == "OK" and proposed:
                src = os.path.join(in_dir, orig)
                dst = os.path.join(in_dir, proposed)
                # Avoid overwrite
                i = 2
                base, ext = os.path.splitext(dst)
                while os.path.exists(dst):
                    dst = f"{base} ({i}){ext}"
                    i += 1
                try:
                    os.replace(src, dst)
                except Exception as e:
                    print(f"[WARN] Nelze přejmenovat {orig}: {e}")
    print(f"Hotovo. Log: {log_path}")
    print("Tip: zkontroluj log a když je to v pohodě, spusť znovu s --inplace.")

if __name__ == "__main__":
    main()
