from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, Response

from openai import OpenAI

import os
import io
import uuid
import tempfile
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from textwrap import wrap
from typing import Optional

import imageio_ffmpeg as ffmpeg
from dotenv import load_dotenv
import stripe
from supabase import create_client, Client
import jwt  # PyJWT
import requests

from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# -------------------------
# Env / Clients
# -------------------------
load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not SUPABASE_JWT_SECRET:
    raise RuntimeError("Missing required SUPABASE_* env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

LULU_CLIENT_KEY = os.getenv("LULU_CLIENT_KEY")
LULU_CLIENT_SECRET = os.getenv("LULU_CLIENT_SECRET")
LULU_BASE_URL = os.getenv("LULU_BASE_URL", "https://api.sandbox.lulu.com")
LULU_POD_PACKAGE_ID = os.getenv("LULU_POD_PACKAGE_ID")
LULU_CONTACT_EMAIL = os.getenv("LULU_CONTACT_EMAIL", "you@yourdomain.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://speech-to-text-o5lh.onrender.com",
        "https://booksly.co",
        "https://www.booksly.co",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FFMPEG = ffmpeg.get_ffmpeg_exe()
DOMAIN = "https://www.booksly.co"

# In-memory results cache (prototype)
results: dict[str, dict] = {}

MAX_TRANSCRIPTION_WORKERS = 8


# -------------------------
# Lulu sizing constants
# -------------------------
TRIM_W_IN = 4.25
TRIM_H_IN = 6.875
BLEED_IN = 0.125

LULU_INTERIOR_PAGE_SIZE = (TRIM_W_IN * inch, TRIM_H_IN * inch)
LULU_COVER_SPREAD_SIZE = ((2 * TRIM_W_IN + 2 * BLEED_IN) * inch, (TRIM_H_IN + 2 * BLEED_IN) * inch)


# -------------------------
# Auth / CORS preflight
# -------------------------
def verify_token(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth header")

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return str(sub)
    except HTTPException:
        raise
    except Exception as e:
        print("❌ JWT decode failed:", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(status_code=200)


# -------------------------
# Fonts (EMBEDDED)
# -------------------------
def _try_register_font_pair(regular_path: str, bold_path: str, regular_name: str, bold_name: str) -> tuple[str, str]:
    if os.path.exists(regular_path) and os.path.exists(bold_path):
        if regular_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(regular_name, regular_path))
        if bold_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(bold_name, bold_path))
        return regular_name, bold_name
    raise FileNotFoundError(f"Missing fonts: {regular_path}, {bold_path}")


def register_embedded_fonts() -> tuple[str, str]:
    """
    Ensure we use embedded TTF fonts so Lulu doesn't reject PDFs.
    Prefer bundled fonts (./fonts) if you add them to your repo,
    otherwise try system fonts.
    """
    # 1) Bundled fonts (recommended)
    bundled = [
        ("./fonts/DejaVuSans.ttf", "./fonts/DejaVuSans-Bold.ttf", "BookifySans", "BookifySans-Bold"),
        ("./fonts/DejaVuSerif.ttf", "./fonts/DejaVuSerif-Bold.ttf", "BookifySerif", "BookifySerif-Bold"),
    ]
    for reg, bold, reg_name, bold_name in bundled:
        try:
            return _try_register_font_pair(reg, bold, reg_name, bold_name)
        except Exception:
            pass

    # 2) Common Linux paths
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  "BookifySans",  "BookifySans-Bold"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "BookifySerif", "BookifySerif-Bold"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "BookifySans", "BookifySans-Bold"),
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", "BookifySerif", "BookifySerif-Bold"),
    ]

    last_err: Exception | None = None
    for reg, bold, reg_name, bold_name in candidates:
        try:
            return _try_register_font_pair(reg, bold, reg_name, bold_name)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "No embeddable TTF fonts found. "
        "Fix: add fonts into ./fonts (recommended) or ensure DejaVu/Liberation fonts exist on the server."
    ) from last_err


FONT_BODY, FONT_BOLD = register_embedded_fonts()


# -------------------------
# Audio helpers
# -------------------------
def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def compress_to_mp3(input_path: str) -> str:
    """Always re-encode to a clean MP3 Whisper can read.
    Uses 16 kHz sample rate because Whisper resamples to 16 kHz internally,
    so higher rates just waste encoding time and bandwidth."""
    output_path = tempfile.mktemp(suffix=".mp3")
    _run_ffmpeg([
        FFMPEG, "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        output_path,
    ])
    return output_path


def get_duration_seconds(file_path: str) -> float:
    """Parse ffmpeg stderr for Duration (works even if ffprobe isn't installed)."""
    proc = subprocess.run([FFMPEG, "-i", file_path], capture_output=True, text=True)
    text = proc.stderr or ""
    import re
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", text)
    if not m:
        return 0.0
    hh = float(m.group(1))
    mm = float(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss


def chunk_audio_mp3(file_path: str, chunk_size_mb: int = 10) -> list[str]:
    """
    Split a (clean) mp3 into multiple smaller mp3s.
    IMPORTANT: re-encode each chunk; DO NOT '-acodec copy'.
    """
    size = os.path.getsize(file_path)
    if size <= chunk_size_mb * 1024 * 1024:
        return [file_path]

    dur = get_duration_seconds(file_path)
    if dur <= 0:
        return [file_path]

    n_parts = int(size // (chunk_size_mb * 1024 * 1024)) + 1
    part_dur = max(30.0, dur / n_parts)

    chunk_paths: list[str] = []
    for i in range(n_parts):
        part_path = tempfile.mktemp(suffix=f"_part{i}.mp3")
        start_time = i * part_dur

        _run_ffmpeg([
            FFMPEG, "-y",
            "-ss", str(start_time),
            "-i", file_path,
            "-t", str(part_dur),
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            part_path,
        ])
        if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
            chunk_paths.append(part_path)

    return chunk_paths or [file_path]


# -------------------------
# PDF generation (INTERIOR + COVER)
# -------------------------
def make_interior_pdf(chapter_text: str, title: str = "User") -> str:
    """Interior PDF must match Lulu trim size; fonts must be embedded."""
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = FONT_BODY
    styles["Normal"].fontSize = 11
    styles["Normal"].leading = 14

    styles["Title"].fontName = FONT_BOLD
    styles["Title"].fontSize = 18
    styles["Title"].leading = 22

    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LULU_INTERIOR_PAGE_SIZE,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 0.2 * inch),
    ]

    for paragraph in chapter_text.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue
        story.append(Paragraph(p, styles["Normal"]))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)
    return pdf_path


def make_cover_pdf(title: str, author: str) -> str:
    """
    Full-spread cover PDF at Lulu cover spread size.
    IMPORTANT: use embedded TTF fonts (not Helvetica) to satisfy Lulu.
    """
    pdf_path = tempfile.mktemp(suffix="_cover.pdf")
    w, h = LULU_COVER_SPREAD_SIZE
    c = canvas.Canvas(pdf_path, pagesize=(w, h))

    # Background
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    # Front cover center (spine assumed ~0)
    front_x0 = (w / 2.0)
    front_center_x = front_x0 + (TRIM_W_IN * inch) / 2.0
    front_center_y = h / 2.0

    c.setFillColorRGB(0, 0, 0)

    c.setFont(FONT_BOLD, 24)
    c.drawCentredString(front_center_x, front_center_y + 40, title[:60])

    # c.setFont(FONT_BODY, 14)
    # c.drawCentredString(front_center_x, front_center_y, f"by {author}"[:80])

    c.save()
    return pdf_path


# -------------------------
# Lulu helpers
# -------------------------
def get_lulu_token() -> str:
    if not LULU_CLIENT_KEY or not LULU_CLIENT_SECRET:
        raise RuntimeError("Missing LULU_CLIENT_KEY/LULU_CLIENT_SECRET")

    token_url = f"{LULU_BASE_URL}/auth/realms/glasstree/protocol/openid-connect/token"
    resp = requests.post(
        token_url,
        data={"grant_type": "client_credentials"},
        auth=(LULU_CLIENT_KEY, LULU_CLIENT_SECRET),
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Lulu token error {resp.status_code}: {resp.text}")
    return resp.json()["access_token"]


def supabase_public_url(bucket: str, path: str) -> str:
    public_res = supabase.storage.from_(bucket).get_public_url(path)
    if isinstance(public_res, str):
        return public_res

    url = (
        getattr(public_res, "public_url", None)
        or getattr(public_res, "publicUrl", None)
        or (public_res.get("publicUrl") if isinstance(public_res, dict) else None)
        or (public_res.get("public_url") if isinstance(public_res, dict) else None)
    )
    if not url:
        raise RuntimeError(f"Could not get public URL for {bucket}/{path}: {public_res}")
    return url


def send_to_printer(interior_storage_path: str, cover_storage_path: str, user_id: str, order_id: str):
    """Requires BOTH interior + cover PDFs."""
    try:
        interior_url = supabase_public_url("book_files", interior_storage_path)
        cover_url = supabase_public_url("book_files", cover_storage_path)
    except Exception as e:
        print(f"❌ Public URL error: {e}")
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    order_row = supabase.table("orders").select("*").eq("id", order_id).single().execute()
    order_data = order_row.data or {}

    shipping_address = {
        "name": order_data.get("ship_name", "Bookify Test User"),
        "street1": order_data.get("ship_line1", "123 Test St"),
        "city": order_data.get("ship_city", "Durham"),
        "state_code": order_data.get("ship_state", "NC"),
        "country_code": order_data.get("ship_country", "US"),
        "postcode": order_data.get("ship_postal", "27701"),
        "phone_number": order_data.get("ship_phone", "+15555555555"),
    }
    if order_data.get("ship_line2"):
        shipping_address["street2"] = order_data["ship_line2"]
    if order_data.get("ship_email"):
        shipping_address["email"] = order_data["ship_email"]

    payload = {
        "contact_email": LULU_CONTACT_EMAIL,
        "shipping_address": shipping_address,
        "shipping_option_level": "MAIL",
        "external_id": str(order_id),
        "line_items": [
            {
                "quantity": 1,
                "title": order_data.get("title", f"Bookify Order {order_id[:8]}"),
                "printable_normalization": {
                    "pod_package_id": LULU_POD_PACKAGE_ID,
                    "cover": {"source_url": cover_url},
                    "interior": {"source_url": interior_url},
                },
            }
        ],
    }

    try:
        token = get_lulu_token()
        resp = requests.post(
            f"{LULU_BASE_URL}/print-jobs/",
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=60,
        )
        print(f"📨 Lulu response status: {resp.status_code}")
        print(f"📨 Lulu response body: {resp.text}")

        if resp.status_code not in (200, 201):
            supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
            return

        job = resp.json()
        lulu_job_id = job.get("id")
        supabase.table("orders").update({
            "status": "Printing",
            "lulu_job_id": str(lulu_job_id) if lulu_job_id else None,
        }).eq("id", order_id).execute()

    except Exception as e:
        print(f"❌ Lulu print-job failed: {e}")
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()


# -------------------------
# Background processing
# -------------------------
def process_audio(upload_id: str, temp_path: str, user_id: str, has_paid: bool, order_id: Optional[str] = None, mode: str = "narrative"):
    print(f"▶️ process_audio START upload_id={upload_id}, has_paid={has_paid}, order_id={order_id}, mode={mode}, temp_path={temp_path}")

    interior_pdf_path = None
    cover_pdf_path = None

    try:
        compressed_path = compress_to_mp3(temp_path)
        chunks = chunk_audio_mp3(compressed_path)

        transcripts: list[str] = []

        def _transcribe_chunk(idx_path: tuple[int, str]) -> tuple[int, str]:
            """Transcribe a single chunk; returns (index, text) for ordering."""
            idx, path = idx_path
            with open(path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return idx, transcript.text

        # Transcribe chunks in parallel – each call is I/O-bound (network),
        # so using threads gives a near-linear speedup.
        with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_TRANSCRIPTION_WORKERS)) as executor:
            futures = {
                executor.submit(_transcribe_chunk, (i, path)): (i, path)
                for i, path in enumerate(chunks)
            }
            indexed_results: list[tuple[int, str]] = []
            try:
                for future in as_completed(futures):
                    indexed_results.append(future.result())
                    _, path = futures[future]
                    if path != compressed_path and os.path.exists(path):
                        os.remove(path)
            finally:
                # Clean up any remaining chunk files on success or failure
                for _, path in futures.values():
                    if path != compressed_path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

        # Reassemble transcripts in original chunk order
        indexed_results.sort(key=lambda x: x[0])
        transcripts = [text for _, text in indexed_results]

        full_text = "\n".join(transcripts)

        if has_paid:
            for p in [temp_path, compressed_path]:
                if p and os.path.exists(p):
                    os.remove(p)

        if mode == "transcription":
            # Pure transcription — return the raw Whisper output with light cleanup
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise transcription formatter. "
                            "Clean up the following raw speech-to-text transcript: fix obvious typos, "
                            "add proper punctuation and paragraph breaks, but do NOT embellish, "
                            "summarize, or change any of the speaker's words. "
                            "Preserve the original meaning and wording exactly."
                        ),
                    },
                    {"role": "user", "content": f"Format this transcript:\n\n{full_text}"},
                ],
                max_tokens=16384,
            )
            chapters_text = completion.choices[0].message.content.strip()
        else:
            # Narrative mode — transform into a structured book
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced editor and storyteller who transforms a long-form spoken transcript "
                            "into a structured book-style narrative divided into chapters. "
                            "Each chapter should cover a substantial portion of the conversation. "
                            "Use rich prose with smooth transitions. "
                            "Each chapter must include a clear title and multiple detailed paragraphs."
                        ),
                    },
                    {"role": "user", "content": f"Divide this transcript into chapters:\n\n{full_text}"},
                ],
                max_tokens=16384,
            )
            chapters_text = completion.choices[0].message.content.strip()

        if not has_paid:
            preview_lines = chapters_text.splitlines()[:20]
            preview_text = "\n".join(preview_lines) + "\n\n[...] Unlock full text with payment."
            results[upload_id] = {"status": "done", "chapters": preview_text, "is_preview": True}
            return

        results[upload_id] = {"status": "done", "chapters": chapters_text, "is_preview": False}

        # Use the custom title from the order if shipping was already submitted
        cover_title = "Booksly Session"
        if order_id:
            title_row = supabase.table("orders").select("title, shipping_submitted").eq("id", order_id).single().execute()
            if title_row.data and title_row.data.get("shipping_submitted") and title_row.data.get("title"):
                cover_title = title_row.data["title"]

        interior_pdf_path = make_interior_pdf(chapters_text, cover_title)
        cover_pdf_path = make_cover_pdf(title=cover_title, author=user_id)

        interior_storage_path = f"books/{user_id}/{upload_id}.pdf"
        cover_storage_path = f"books/{user_id}/{upload_id}_cover.pdf"

        with open(interior_pdf_path, "rb") as f:
            supabase.storage.from_("book_files").upload(interior_storage_path, f)
        with open(cover_pdf_path, "rb") as f:
            supabase.storage.from_("book_files").upload(cover_storage_path, f)

        book_insert = supabase.table("user_books").insert({
            "user_id": user_id,
            "title": cover_title if cover_title != "Booksly Session" else f"Session {upload_id[:8]}",
            "content": chapters_text,
            "pdf_path": interior_storage_path,
        }).execute()

        book_id = book_insert.data[0]["id"] if book_insert.data else None

        if order_id and book_id:
            supabase.table("orders").update({"book_id": book_id}).eq("id", order_id).execute()

            order_row = supabase.table("orders").select("type, shipping_submitted").eq("id", order_id).single().execute()
            if order_row.data and order_row.data.get("type") == "book":
                if order_row.data.get("shipping_submitted"):
                    send_to_printer(interior_storage_path, cover_storage_path, user_id, order_id)
                else:
                    print("⏸ Book ready, waiting for user to submit shipping details.")

        supabase.table("upload_sessions").update({"status": "complete"}).eq("id", upload_id).execute()

    except Exception as e:
        import traceback
        traceback.print_exc()
        results[upload_id] = {"status": "error", "error": str(e)}
        print("❌ Error in process_audio:", e)

    finally:
        for p in [interior_pdf_path, cover_pdf_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -------------------------
# Routes
# -------------------------
@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), authorization: str = Header(None), mode: str = "narrative"):
    user_id = verify_token(authorization)
    upload_id = str(uuid.uuid4())

    if mode not in ("narrative", "transcription"):
        mode = "narrative"

    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.split(".")[-1].lower()
    if not suffix:
        suffix = ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    storage_audio_path = f"uploads/{user_id}/{upload_id}{suffix}"
    with open(temp_path, "rb") as f:
        supabase.storage.from_("raw_audio").upload(storage_audio_path, f)

    supabase.table("upload_sessions").insert({
        "id": upload_id,
        "user_id": user_id,
        "audio_path": storage_audio_path,
        "status": "preview",
        "mode": mode,
    }).execute()

    results[upload_id] = {"status": "processing", "paid": False, "mode": mode}

    threading.Thread(target=process_audio, args=(upload_id, temp_path, user_id, False, None, mode), daemon=True).start()
    return {"id": upload_id, "status": "processing", "mode": mode}


@app.get("/result/{upload_id}")
def get_result(upload_id: str):
    return results.get(upload_id, {"status": "not_found"})


@app.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    data = await request.json()
    purchase_type = data.get("type")
    user_id = data.get("user_id")
    upload_id = data.get("upload_id")

    if purchase_type == "pdf":
        price_id = os.getenv("STRIPE_PRICE_PDF")
        success_url = f"{DOMAIN}/pdf-download"
    elif purchase_type == "book":
        price_id = os.getenv("STRIPE_PRICE_BOOK")
        success_url = f"{DOMAIN}/book-customize"
    else:
        return JSONResponse({"error": "Invalid type"}, status_code=400)

    if not price_id:
        return JSONResponse({"error": "Missing Stripe price id env var"}, status_code=500)

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="payment",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=f"{DOMAIN}/upload",
        metadata={
            "user_id": user_id,
            "upload_id": upload_id,
            "order_type": purchase_type,
            "title": "Booksly Order",
        },
    )
    return {"url": session.url}


@app.post("/submit-shipping")
async def submit_shipping(request: Request, user_id: str = Depends(verify_token)):
    data = await request.json()
    order_id = data.get("order_id")
    custom_title = data.get("custom_title")
    shipping = data.get("shipping", {})

    if not order_id:
        raise HTTPException(status_code=400, detail="Missing order_id")

    if not custom_title or not custom_title.strip():
        raise HTTPException(status_code=400, detail="Missing custom_title")

    custom_title = custom_title.strip()

    required = ["name", "street1", "city", "state_code", "country_code", "postcode"]
    for field in required:
        if not shipping.get(field):
            raise HTTPException(status_code=400, detail=f"Missing required shipping field: {field}")

    order_res = (
        supabase.table("orders")
        .select("*")
        .eq("id", order_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not order_res.data:
        raise HTTPException(status_code=404, detail="Order not found")

    supabase.table("orders").update({
        "title": custom_title,
        "ship_name": shipping.get("name", ""),
        "ship_email": shipping.get("email", ""),
        "ship_phone": shipping.get("phone_number", ""),
        "ship_line1": shipping.get("street1", ""),
        "ship_line2": shipping.get("street2", ""),
        "ship_city": shipping.get("city", ""),
        "ship_state": shipping.get("state_code", ""),
        "ship_country": shipping.get("country_code", ""),
        "ship_postal": shipping.get("postcode", ""),
        "shipping_submitted": True,
    }).eq("id", order_id).execute()

    # If book already exists, update its title, regenerate cover, and print now
    order_data = order_res.data
    if order_data.get("type") == "book" and order_data.get("book_id"):
        supabase.table("user_books").update({
            "title": custom_title,
        }).eq("id", order_data["book_id"]).execute()

        book_res = supabase.table("user_books").select("pdf_path").eq("id", order_data["book_id"]).single().execute()
        if book_res.data and book_res.data.get("pdf_path"):
            interior_path = book_res.data["pdf_path"]
            cover_path = interior_path.replace(".pdf", "_cover.pdf")

            # Regenerate the cover PDF with the user's custom title
            new_cover_pdf = None
            try:
                new_cover_pdf = make_cover_pdf(title=custom_title, author=user_id)
                with open(new_cover_pdf, "rb") as f:
                    supabase.storage.from_("book_files").update(cover_path, f)
            except Exception as e:
                print(f"⚠️ Could not regenerate cover with custom title: {e}")
            finally:
                if new_cover_pdf and os.path.exists(new_cover_pdf):
                    try:
                        os.remove(new_cover_pdf)
                    except Exception:
                        pass

            try:
                _ = supabase_public_url("book_files", cover_path)
                threading.Thread(target=send_to_printer, args=(interior_path, cover_path, user_id, order_id), daemon=True).start()
            except Exception:
                print("⏸ Shipping saved, but cover PDF not available yet.")

    return {"status": "ok"}


@app.get("/orders/{user_id}")
async def get_orders(user_id: str):
    res = supabase.table("orders").select("*").eq("user_id", user_id).execute()
    return res.data


@app.get("/order-from-session")
async def order_from_session(session_id: str, user_id: str = Depends(verify_token)):
    res = (
        supabase.table("orders")
        .select("id")
        .eq("stripe_session_id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Order not found for this session (may still be processing).")
    return {"order_id": res.data[0]["id"]}



@app.get("/download-latest-pdf")
def download_latest_pdf(user_id: str = Depends(verify_token)):
    # Find latest book row for user (must include pdf_path)
    res = (
        supabase.table("user_books")
        .select("pdf_path, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        raise HTTPException(status_code=404, detail="No book found for this user.")
    pdf_path = res.data[0].get("pdf_path")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="No PDF available for this user.")

    # Download the PDF bytes from Supabase Storage
    file_data = supabase.storage.from_("book_files").download(pdf_path)
    pdf_bytes = getattr(file_data, "content", file_data)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="SpeechToBook.pdf"'},
    )

@app.get("/download/{order_id}")
def download_order_pdf(order_id: str, user_id: str = Depends(verify_token)):
    # Find book_id for this order
    order_res = (
        supabase.table("orders")
        .select("book_id")
        .eq("id", order_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    book_id = None
    if order_res.data:
        book_id = order_res.data[0].get("book_id")

    # Fallback: latest book if order not linked yet
    if not book_id:
        latest = (
            supabase.table("user_books")
            .select("id")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if latest.data:
            book_id = latest.data[0]["id"]

    if not book_id:
        raise HTTPException(status_code=404, detail="No book available for this order.")

    book_res = (
        supabase.table("user_books")
        .select("pdf_path")
        .eq("id", book_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not book_res.data or not book_res.data[0].get("pdf_path"):
        raise HTTPException(status_code=404, detail="PDF not found for this order.")

    pdf_path = book_res.data[0]["pdf_path"]
    file_data = supabase.storage.from_("book_files").download(pdf_path)
    pdf_bytes = getattr(file_data, "content", file_data)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="SpeechToBook.pdf"'},
    )

@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        stripe_session_id = session.get("id")
        meta = session.get("metadata", {}) or {}
        user_id = meta.get("user_id")
        upload_id = meta.get("upload_id")
        order_type = meta.get("order_type") or "pdf"
        title = "Booksly Order"
        # title = meta.get("title") or "Bookify Order"

        status_order = "Complete" if order_type == "pdf" else "Processing"
        order_insert = supabase.table("orders").insert({
            "user_id": user_id,
            "title": title,
            "type": order_type,
            "status": status_order,
            "stripe_session_id": stripe_session_id,
        }).execute()
        order_id = order_insert.data[0]["id"] if order_insert.data else None

        upload_row = supabase.table("upload_sessions").select("audio_path, mode").eq("id", upload_id).single().execute()
        audio_path = upload_row.data["audio_path"] if upload_row.data else None
        upload_mode = (upload_row.data.get("mode") or "narrative") if upload_row.data else "narrative"
        if not audio_path:
            return JSONResponse(status_code=200, content={"status": "missing audio_path"})

        file_data = supabase.storage.from_("raw_audio").download(audio_path)
        suffix = os.path.splitext(audio_path)[1] or ".bin"
        temp_path = tempfile.mktemp(suffix=suffix)
        with open(temp_path, "wb") as f:
            f.write(getattr(file_data, "content", file_data))

        threading.Thread(target=process_audio, args=(upload_id, temp_path, user_id, True, order_id, upload_mode), daemon=True).start()
        supabase.table("upload_sessions").update({"status": "processing"}).eq("id", upload_id).execute()

    return JSONResponse(status_code=200, content={"status": "success"})


if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
