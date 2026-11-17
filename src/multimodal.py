# import base64
# import hashlib
# import json
# import re
# from io import BytesIO
# from pathlib import Path
# from urllib.error import HTTPError, URLError
# from urllib.request import Request, urlopen

# import pymupdf
# from PIL import Image

# OLLAMA_URL = "http://localhost:11434/api/generate"
# VLM_MODEL = "qwen3-vl:4b-instruct"
# MIN_IMAGE_PX = 150
# MAX_IMAGE_DIM = 1000

# VLM_PROMPT = """Extract the content of this image from a medical document.

# If it is a table: output the title, then a Markdown table with every header, row, cell value and unit exactly as shown, then any footnotes. Empty cell for blanks. Do not skip or merge rows.
# If it is a chart or figure: output the title, axis labels with units, each series with its values (Markdown table where possible), and one or two sentences on the main relationship.
# If it is only a logo, photo, icon or decoration with no data: output exactly SKIP.

# Write [unreadable] for illegible text. Do not add anything not in the image. Return only the extracted content."""

# # cache key changes when model or prompt changes
# PROMPT_HASH = hashlib.sha1((VLM_MODEL + VLM_PROMPT).encode()).hexdigest()[:8]
# MD_SEPARATOR = re.compile(r"^\|[\s:\-|]*-{3,}", re.M)


# def _to_jpeg_b64(image_bytes, max_dim=MAX_IMAGE_DIM):
#     with Image.open(BytesIO(image_bytes)) as im:
#         im = im.convert("RGB")
#         im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
#         buf = BytesIO()
#         im.save(buf, format="JPEG", quality=85, optimize=True)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# def _vlm_extract(image_bytes):
#     """Returns (text, complete). complete=False if the output hit num_predict."""
#     payload = {
#         "model": VLM_MODEL,
#         "prompt": VLM_PROMPT,
#         "images": [_to_jpeg_b64(image_bytes)],
#         "stream": True,
#         "think": False,
#         "keep_alive": "5m",
#         "options": {
#             "num_ctx": 16384,
#             "num_predict": 3000,
#             "temperature": 0,
#             "repeat_penalty": 1.1,
#         },
#     }
#     request = Request(
#         OLLAMA_URL,
#         data=json.dumps(payload).encode("utf-8"),
#         headers={"Content-Type": "application/json"},
#     )
#     chunks, complete = [], True
#     with urlopen(request, timeout=300) as response:   # idle-gap timeout
#         for line in response:
#             msg = json.loads(line)
#             chunks.append(msg.get("response", ""))
#             if msg.get("done"):
#                 complete = msg.get("done_reason") == "stop"
#                 break
#     return "".join(chunks).strip(), complete


# def _table_to_markdown(rows):
#     rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if any(r)]
#     if len(rows) < 2:
#         return ""
#     header, body = rows[0], rows[1:]
#     lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
#     lines += ["| " + " | ".join(r) + " |" for r in body]
#     return "\n".join(lines)


# def _load_cache(path):
#     return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# def _save_cache(path, cache):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# def load_pdf_multimodal(file_path, cache_dir="cache"):
#     """
#     Returns elements: {"id", "text", "metadata": {source, page, type, origin}}
#     type   = text | table | image
#     origin = pdf_text | native_table | vlm
#     """
#     path = Path(file_path)
#     if not path.exists():
#         raise FileNotFoundError(f"PDF not found: {file_path}")

#     cache_path = Path(cache_dir) / f"{path.stem}.vlm.json"
#     cache = _load_cache(cache_path)
#     seen_images = set()
#     elements = []

#     doc = pymupdf.open(str(path))
#     try:
#         for page in doc:
#             page_no = page.number + 1
#             idx = 0

#             def add(kind, origin, text):
#                 nonlocal idx
#                 elements.append({
#                     "id": f"{path.stem}_p{page_no}_{kind}_{idx}",
#                     "text": text,
#                     "metadata": {"source": path.name, "page": page_no,
#                                  "type": kind, "origin": origin},
#                 })
#                 idx += 1

#             # 1. native tables (no VLM)
#             table_rects = []
#             for t in page.find_tables().tables:
#                 if t.row_count < 2 or t.col_count < 2:   # filter false positives
#                     continue
#                 md = _table_to_markdown(t.extract())
#                 if md:
#                     table_rects.append(pymupdf.Rect(t.bbox))
#                     add("table", "native_table", md)

#             # 2. body text, excluding text already captured inside native tables
#             parts = []
#             for x0, y0, x1, y1, txt, _, btype in page.get_text("blocks", sort=True):
#                 if btype != 0:
#                     continue
#                 center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
#                 if any(r.contains(center) for r in table_rects):
#                     continue
#                 if txt.strip():
#                     parts.append(txt.strip())
#             if parts:
#                 add("text", "pdf_text", "\n".join(parts))

#             # 3. raster images -> VLM (cached)
#             for img in page.get_images(full=True):
#                 xref, w, h = img[0], img[2], img[3]
#                 if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
#                     continue
#                 rects = page.get_image_rects(xref)
#                 if rects and any(tr.contains(r) for r in rects for tr in table_rects):
#                     continue

#                 raw = doc.extract_image(xref)["image"]
#                 digest = hashlib.sha1(raw).hexdigest()
#                 if digest in seen_images:      # repeated logo/header image
#                     continue
#                 seen_images.add(digest)

#                 key = f"{digest}:{PROMPT_HASH}"
#                 content = cache.get(key)
#                 if content is None:
#                     print(f"VLM: page {page_no}, xref {xref}")
#                     try:
#                         content, complete = _vlm_extract(raw)
#                     except (HTTPError, URLError, TimeoutError, OSError) as e:
#                         print(f"  skipped (VLM error): {e}")
#                         continue
#                     if complete:               # never cache truncated output
#                         cache[key] = content
#                         _save_cache(cache_path, cache)

#                 if not content or content.upper().startswith("SKIP"):
#                     continue
#                 kind = "table" if MD_SEPARATOR.search(content) else "image"
#                 add(kind, "vlm", content)
#     finally:
#         doc.close()

#     return elements

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import pymupdf
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen3-vl:4b-instruct"
MIN_IMAGE_PX = 150

VLM_PROMPT = """Extract the content of this image from a medical document.

If it is a table: output the title, then a Markdown table with every header, row, cell value and unit exactly as shown, then any footnotes. Empty cell for blanks. Do not skip or merge rows.
If it is a chart or figure: output the title, axis labels with units, each series with its values (Markdown table where possible), and one or two sentences on the main relationship.
If it is only a logo, photo, icon or decoration with no data: output exactly SKIP.

Write [unreadable] for illegible text. Do not add anything not in the image. Return only the extracted content."""

PROMPT_HASH = hashlib.sha1((VLM_MODEL + VLM_PROMPT).encode()).hexdigest()[:8]


def vlm_extract(image_bytes):
    """Returns (text, complete). complete=False if output hit num_predict."""
    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        im.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)

    payload = {
        "model": VLM_MODEL,
        "prompt": VLM_PROMPT,
        "images": [base64.b64encode(buf.getvalue()).decode()],
        "stream": True,
        "think": False,
        "keep_alive": "5m",
        "options": {"num_ctx": 16384, "num_predict": 3000,
                    "temperature": 0, "repeat_penalty": 1.1},
    }
    request = Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})

    parts, complete = [], True
    with urlopen(request, timeout=300) as response:   # idle-gap timeout
        for line in response:
            msg = json.loads(line)
            parts.append(msg.get("response", ""))
            if msg.get("done"):
                complete = msg.get("done_reason") == "stop"
                break
    return "".join(parts).strip(), complete


def load_pdf_multimodal(file_path, cache_dir="cache"):
    """Returns elements: {"id", "text", "metadata": {source, page, type}}, type = text | image."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    cache_path = Path(cache_dir) / f"{path.stem}.vlm.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    elements = []

    def add(page_no, kind, text):
        elements.append({
            "id": f"{path.stem}_p{page_no}_{len(elements)}",
            "text": text,
            "metadata": {"source": path.name, "page": page_no, "type": kind},
        })

    doc = pymupdf.open(str(path))
    try:
        for page in doc:
            n = page.number + 1

            text = page.get_text("text").strip()
            if text:
                add(n, "text", text)

            for img in page.get_images(full=True):
                xref, w, h = img[0], img[2], img[3]
                if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
                    continue

                raw = doc.extract_image(xref)["image"]
                key = f"{hashlib.sha1(raw).hexdigest()}:{PROMPT_HASH}"

                content = cache.get(key)
                if content is None:
                    print(f"VLM: page {n}, xref {xref}")
                    try:
                        content, complete = vlm_extract(raw)
                    except OSError as e:      # covers URLError, HTTPError, TimeoutError
                        print(f"  skipped: {e}")
                        continue
                    if complete:              # never cache truncated output
                        cache[key] = content

                if content and len(content) >= 150 and not content.upper().startswith("SKIP"):
                    add(n, "image", content)
    finally:
        doc.close()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    return elements