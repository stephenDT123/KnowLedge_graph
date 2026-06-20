from __future__ import annotations

import json
import io
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Set, Union

try:
    import fitz
except Exception:
    fitz = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import jieba
except Exception:
    jieba = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None


def load_stopwords(stopwords_path: Optional[Union[str, os.PathLike]] = None) -> Set[str]:
    if stopwords_path is None:
        stopwords_path = Path(__file__).with_name("stopwords-zh.json")
    try:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(str(x).strip() for x in data if str(x).strip())
        if isinstance(data, dict):
            return set(str(x).strip() for x in data.keys() if str(x).strip())
    except Exception:
        pass
    return {"的", "是", "在", "和", "了", "等"}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Union[str, os.PathLike], prefer_pymupdf: bool = True) -> str:
    pdf_path = str(pdf_path)
    if prefer_pymupdf and fitz is not None:
        doc = fitz.open(pdf_path)
        parts: List[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        doc.close()
        return "\n".join(parts)

    if PyPDF2 is None:
        raise ModuleNotFoundError("缺少PDF解析依赖：请安装 PyMuPDF 或 PyPDF2")

    parts: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def ocr_images_from_pdf(
    pdf_path: Union[str, os.PathLike],
    lang: str = "chi_sim",
    tesseract_cmd: Optional[str] = None,
    max_images_per_page: int = 8,
) -> str:
    if fitz is None:
        raise ModuleNotFoundError("OCR需要安装 PyMuPDF（pymupdf）")
    if pytesseract is None:
        raise ModuleNotFoundError("OCR需要安装 pytesseract")
    if Image is None:
        raise ModuleNotFoundError("OCR需要安装 pillow")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if shutil.which("tesseract") is None and not tesseract_cmd:
        raise RuntimeError("未检测到tesseract可执行文件，请安装Tesseract并加入PATH，或传入tesseract_cmd")

    doc = fitz.open(str(pdf_path))
    texts: List[str] = []
    for page in doc:
        image_list = page.get_images(full=True)
        for img in image_list[:max_images_per_page]:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image.get("image")
            if not image_bytes:
                continue
            pil_img = Image.open(io.BytesIO(image_bytes))
            texts.append(pytesseract.image_to_string(pil_img, lang=lang) or "")
    doc.close()
    return "\n".join(texts)


def tokenize_zh(text: str) -> List[str]:
    if jieba is None:
        raise ModuleNotFoundError("缺少分词依赖：请安装 jieba")
    return [t.strip() for t in jieba.cut(text) if t and t.strip()]


def filter_tokens(tokens: List[str], stopwords: Set[str]) -> List[str]:
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def process_pdf(
    pdf_path: Union[str, os.PathLike],
    stopwords_path: Optional[Union[str, os.PathLike]] = None,
    enable_ocr: bool = False,
    tesseract_cmd: Optional[str] = None,
) -> dict:
    raw_text = extract_text_from_pdf(pdf_path, prefer_pymupdf=True)
    if enable_ocr:
        try:
            raw_text = raw_text + "\n" + ocr_images_from_pdf(pdf_path, tesseract_cmd=tesseract_cmd)
        except Exception:
            pass

    raw_text = normalize_text(raw_text)
    stopwords = load_stopwords(stopwords_path)
    tokens = tokenize_zh(raw_text)
    filtered = filter_tokens(tokens, stopwords)
    return {
        "pdf_path": str(pdf_path),
        "raw_text": raw_text,
        "tokens": tokens,
        "filtered_tokens": filtered,
        "filtered_text": " ".join(filtered),
    }


if __name__ == "__main__":
    pdf_path = Path("附件：学员操作手册.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到PDF文件：{pdf_path.resolve()}")
    result = process_pdf(pdf_path, enable_ocr=False)
    print(result["filtered_text"])

     # 保存处理后的文本到文件，每隔100个字符换行
    with open("processed_text.txt", "w", encoding="utf-8") as f:
        for i in range(0, len(result["filtered_text"]), 100):
            f.write(result["filtered_text"][i:i+100] + "\n")
