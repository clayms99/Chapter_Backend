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
from reportlab.lib.pagesizes import letter, LETTER
from reportlab.lib.units import inch


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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FFMPEG = ffmpeg.get_ffmpeg_exe()

# In-memory results cache (ok for prototype; move to DB/Redis later)
results: dict[str, dict] = {}

DOMAIN = "https://speech-to-text-o5lh.onrender.com"


# -------------------------
# Lulu sizing constants
# -------------------------
TRIM_W_IN = 4.25
TRIM_H_IN = 6.875
BLEED_IN = 0.125

# Interior page size for Lulu trim
LULU_INTERIOR_PAGE_SIZE = (TRIM_W_IN * inch, TRIM_H_IN * inch)

# Cover spread for thin books (spine ~ 0 for very low page counts)
# width = back + front + bleed left+right = 2*trim_w + 2*bleed (spine=0)
# height = trim_h + 2*bleed
LULU_COVER_SPREAD_SIZE = ((2 * TRIM_W_IN + 2 * BLEED_IN) * inch, (TRIM_H_IN + 2 * BLEED_IN) * inch)


# -------------------------
# Helpers
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
        print("✅ TOKEN PAYLOAD:", payload)
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


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def compress_to_mp3(input_path: str) -> str:
    """
    Always re-encode to a clean MP3 Whisper can read.
    """
    output_path = tempfile.mktemp(suffix=".mp3")
    _run_ffmpeg([
        FFMPEG, "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "44100",
        "-b:a", "64k",
        output_path,
    ])
    return output_path


def get_duration_seconds(file_path: str) -> float:
    """
    Use ffmpeg (not ffprobe) to avoid missing ffprobe in some environments.
    Parses stderr output for "Duration:".
    """
    # This is a bit hacky but very reliable without ffprobe availability.
    proc = subprocess.run([FFMPEG, "-i", file_path], capture_output=True, text=True)
    text = proc.stderr or ""
    # Example: Duration: 00:01:23.45,
    import re
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", text)
    if not m:
        # if we can't read duration, just skip chunking
        return 0.0
    hh = float(m.group(1))
    mm = float(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss


def chunk_audio_mp3(file_path: str, chunk_size_mb: int = 10) -> list[str]:
    """
    Split a (clean) mp3 into multiple smaller mp3s.
    IMPORTANT: re-encode each chunk; DO NOT '-acodec copy' (can produce invalid chunks).
    """
    size = os.path.getsize(file_path)
    if size <= chunk_size_mb * 1024 * 1024:
        return [file_path]

    dur = get_duration_seconds(file_path)
    if dur <= 0:
        # If duration couldn't be read, just return original and let Whisper try.
        return [file_path]

    n_parts = int(size // (chunk_size_mb * 1024 * 1024)) + 1
    part_dur = max(30.0, dur / n_parts)  # keep chunks sane

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
            "-ar", "44100",
            "-b:a", "64k",
            part_path,
        ])
        # Ensure not empty
        if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
            chunk_paths.append(part_path)

    # Fallback: if something went wrong, use the original
    return chunk_paths or [file_path]


def make_interior_pdf(chapter_text: str, user_name: str = "User") -> str:
    """
    Interior PDF must match Lulu trim size.
    """
    styles = getSampleStyleSheet()
    pdf_path = tempfile.mktemp(suffix=".pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=LULU_INTERIOR_PAGE_SIZE,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story = []
    story.append(Paragraph(f"<b>{user_name}'s Book</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

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
    Generate a simple full-spread cover PDF at Lulu cover spread size
    (back + front + bleed, spine assumed ~0 for thin books).
    """
    pdf_path = tempfile.mktemp(suffix="_cover.pdf")
    w, h = LULU_COVER_SPREAD_SIZE
    c = canvas.Canvas(pdf_path, pagesize=(w, h))

    # Background (white) - you can change later
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    # Compute front cover area: right half excluding bleed margins
    # Front trim starts at (w/2 + BLEED) when spine=0
    bleed = BLEED_IN * inch
    front_x0 = (w / 2.0)
    front_center_x = front_x0 + (TRIM_W_IN * inch) / 2.0
    front_center_y = h / 2.0

    # Title/author
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(front_center_x, front_center_y + 40, title[:60])

    c.setFont("Helvetica", 14)
    c.drawCentredString(front_center_x, front_center_y, f"by {author}"[:80])

    # Optional: small spine text if you later compute spine
    c.save()
    return pdf_path


def get_lulu_token() -> str:
    if not LULU_CLIENT_KEY or not LULU_CLIENT_SECRET:
        raise RuntimeError("Missing LULU_CLIENT_KEY/LULU_CLIENT_SECRET")

    token_url = f"{LULU_BASE_URL}/auth/realms/glasstree/protocol/openid-connect/token"
    data = {"grant_type": "client_credentials"}

    resp = requests.post(
        token_url,
        data=data,
        auth=(LULU_CLIENT_KEY, LULU_CLIENT_SECRET),
        timeout=20,
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
    """
    Requires BOTH interior + cover PDFs.
    """
    try:
        interior_url = supabase_public_url("book_files", interior_storage_path)
        cover_url = supabase_public_url("book_files", cover_storage_path)
    except Exception as e:
        print(f"❌ Public URL error: {e}")
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    print(f"🌐 Lulu will fetch interior PDF from {interior_url}")
    print(f"🌐 Lulu will fetch cover PDF from {cover_url}")

    order_row = (
        supabase.table("orders")
        .select("*")
        .eq("id", order_id)
        .single()
        .execute()
    )
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
    except Exception as e:
        print(f"❌ Failed to get Lulu token: {e}")
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("📦 Creating Lulu Print-Job...")
    try:
        resp = requests.post(
            f"{LULU_BASE_URL}/print-jobs/",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except Exception as e:
        print(f"❌ Lulu print-job request failed: {e}")
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    print(f"📨 Lulu response status: {resp.status_code}")
    print(f"📨 Lulu response body: {resp.text}")

    if resp.status_code not in (200, 201):
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    job = resp.json()
    lulu_job_id = job.get("id")

    print(f"✅ Lulu Print-Job created: {lulu_job_id}")
    supabase.table("orders").update({
        "status": "Printing",
        "lulu_job_id": str(lulu_job_id) if lulu_job_id else None,
    }).eq("id", order_id).execute()


# -------------------------
# Background processing
# -------------------------
def process_audio(upload_id: str, temp_path: str, user_id: str, has_paid: bool, order_id: Optional[str] = None):
    print(f"▶️ process_audio START upload_id={upload_id}, has_paid={has_paid}, order_id={order_id}, temp_path={temp_path}")

    interior_pdf_path = None
    cover_pdf_path = None

    try:
        # Always convert to clean mp3 first (input could be webm/m4a/etc)
        compressed_path = compress_to_mp3(temp_path)

        # Chunk safely (re-encode chunks)
        chunks = chunk_audio_mp3(compressed_path)
        transcripts: list[str] = []

        for i, path in enumerate(chunks):
            with open(path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            transcripts.append(transcript.text)

            # Remove chunk if it’s not the original compressed file
            if path != compressed_path and os.path.exists(path):
                os.remove(path)

        full_text = "\n".join(transcripts)

        # Clean up original temp files for paid runs
        if has_paid:
            for p in [temp_path, compressed_path]:
                if p and os.path.exists(p):
                    os.remove(p)
        else:
            print(f"⏸ Keeping temp file for preview {upload_id} to allow reprocess later.")

        # GPT chaptering
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced editor and storyteller who transforms a long-form spoken transcript "
                        "into a structured book-style narrative divided into chapters. "
                        "Each chapter should cover a substantial portion of the conversation—typically several related ideas or stories—"
                        "not just short topic shifts. "
                        "Aim for rich, continuous prose with detailed paragraphs that flow naturally. "
                        "The number of chapters should depend on the transcript length: fewer chapters for short transcripts, "
                        "and more chapters for longer ones. "
                        "Each chapter must include a clear, engaging title and multiple detailed paragraphs (typically 5–12) "
                        "that capture the key moments, transitions, and emotions of that section. "
                        "Please write with natural transitions, avoiding repetitive introductions like 'In this section...' "
                        "and instead use smooth storytelling flow."
                    ),
                },
                {"role": "user", "content": f"Divide this transcript into chapters:\n\n{full_text}"},
            ],
            max_tokens=4000,
        )

        chapters_text = completion.choices[0].message.content.strip()
        print(f"✅ Whisper + GPT done for upload_id={upload_id}, has_paid={has_paid}")

        if not has_paid:
            preview_lines = chapters_text.splitlines()[:20]
            preview_text = "\n".join(preview_lines) + "\n\n[...] Unlock full text with payment."
            results[upload_id] = {"status": "done", "chapters": preview_text, "is_preview": True}
            return

        # Paid result
        results[upload_id] = {"status": "done", "chapters": chapters_text, "is_preview": False}

        # Create interior PDF (correct trim)
        interior_pdf_path = make_interior_pdf(chapters_text, user_id)
        print(f"✅ Created interior PDF at {interior_pdf_path}")

        # Create cover PDF (correct spread size)
        cover_pdf_path = make_cover_pdf(title=f"Bookify Session {upload_id[:8]}", author=user_id)
        print(f"✅ Created cover PDF at {cover_pdf_path}")

        # Upload PDFs
        interior_storage_path = f"books/{user_id}/{upload_id}.pdf"
        cover_storage_path = f"books/{user_id}/{upload_id}_cover.pdf"

        with open(interior_pdf_path, "rb") as f:
            supabase.storage.from_("book_files").upload(interior_storage_path, f)
        with open(cover_pdf_path, "rb") as f:
            supabase.storage.from_("book_files").upload(cover_storage_path, f)

        print(f"✅ Uploaded interior PDF to Supabase: {interior_storage_path}")
        print(f"✅ Uploaded cover PDF to Supabase: {cover_storage_path}")

        # Save to user_books
        book_insert = supabase.table("user_books").insert({
            "user_id": user_id,
            "title": f"Session {upload_id[:8]}",
            "content": chapters_text,
            "pdf_path": interior_storage_path,
        }).execute()

        book_id = book_insert.data[0]["id"] if book_insert.data else None
        if book_id:
            print(f"✅ Saved book {book_id} for user {user_id}")

        # Link to order + print if ready
        if order_id and book_id:
            supabase.table("orders").update({"book_id": book_id}).eq("id", order_id).execute()
            print(f"✅ Linked book {book_id} → order {order_id}")

            order_data = (
                supabase.table("orders")
                .select("type, shipping_submitted")
                .eq("id", order_id)
                .single()
                .execute()
            )

            if order_data.data and order_data.data.get("type") == "book":
                if order_data.data.get("shipping_submitted"):
                    print("🚀 Shipping already submitted — sending to Lulu now...")
                    send_to_printer(interior_storage_path, cover_storage_path, user_id, order_id)
                else:
                    print("⏸ Book ready, waiting for user to submit shipping details.")

        # Mark upload complete
        try:
            supabase.table("upload_sessions").update({"status": "complete"}).eq("id", upload_id).execute()
            print(f"🏁 upload_sessions status updated to 'complete' for {upload_id}")
        except Exception as e:
            print(f"⚠️ Failed to update upload_sessions status for {upload_id}: {e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        results[upload_id] = {"status": "error", "error": str(e)}
        print("❌ Error in process_audio:", e)

    finally:
        # local temp cleanup
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
async def upload_audio(
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    user_id = verify_token(authorization)
    upload_id = str(uuid.uuid4())

    # Keep original extension if present; otherwise .bin
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.split(".")[-1].lower()
    if not suffix:
        suffix = ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    # Save raw upload
    storage_audio_path = f"uploads/{user_id}/{upload_id}{suffix}"
    with open(temp_path, "rb") as f:
        supabase.storage.from_("raw_audio").upload(storage_audio_path, f)

    print(f"✅ Uploaded raw audio to Supabase: {storage_audio_path}")

    supabase.table("upload_sessions").insert({
        "id": upload_id,
        "user_id": user_id,
        "audio_path": storage_audio_path,
        "status": "preview",
    }).execute()

    results[upload_id] = {"status": "processing", "paid": False}

    threading.Thread(
        target=process_audio,
        args=(upload_id, temp_path, user_id, False),
        daemon=True,
    ).start()

    return {"id": upload_id, "status": "processing"}


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
            "title": "Bookify Order",
        },
    )

    return {"url": session.url}


@app.get("/download/{order_id}")
async def download_order_pdf(order_id: str, authorization: str = Header(None)):
    user_id = verify_token(authorization)
    print(f"✅ Authenticated PDF download for user {user_id}, order {order_id}")

    order_res = (
        supabase.table("orders")
        .select("book_id")
        .eq("id", order_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    book_id = order_res.data[0]["book_id"] if (order_res.data and order_res.data[0].get("book_id")) else None
    if not book_id:
        raise HTTPException(status_code=404, detail="No book linked to this order.")

    book_res = (
        supabase.table("user_books")
        .select("content")
        .eq("id", book_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not book_res.data:
        raise HTTPException(status_code=404, detail="Book not found for this order.")

    book_text = book_res.data[0]["content"]

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    textobject = p.beginText()
    textobject.setTextOrigin(inch, height - inch)
    textobject.setFont("Helvetica", 12)

    for paragraph in book_text.split("\n"):
        for line in wrap(paragraph, 90):
            textobject.textLine(line)
        textobject.textLine("")
        if textobject.getY() <= inch:
            p.drawText(textobject)
            p.showPage()
            textobject = p.beginText()
            textobject.setTextOrigin(inch, height - inch)
            textobject.setFont("Helvetica", 12)

    p.drawText(textobject)
    p.save()
    buffer.seek(0)

    headers = {"Content-Disposition": "attachment; filename=Bookify.pdf"}
    return StreamingResponse(buffer, headers=headers, media_type="application/pdf")


@app.post("/submit-shipping")
async def submit_shipping(request: Request, user_id: str = Depends(verify_token)):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    order_id = data.get("order_id")
    shipping = data.get("shipping", {})

    if not order_id:
        raise HTTPException(status_code=400, detail="Missing order_id")

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

    print(f"✅ Shipping details saved for order {order_id}")

    # If book already exists, print now
    order_data = order_res.data
    if order_data.get("type") == "book" and order_data.get("book_id"):
        book_res = (
            supabase.table("user_books")
            .select("pdf_path")
            .eq("id", order_data["book_id"])
            .single()
            .execute()
        )

        if book_res.data and book_res.data.get("pdf_path"):
            interior_path = book_res.data["pdf_path"]
            cover_path = interior_path.replace(".pdf", "_cover.pdf")

            # If cover is missing (older books), we can’t print yet
            # (you can regenerate later, but for now just fail gracefully)
            try:
                # verify cover exists by trying to make URL
                _ = supabase_public_url("book_files", cover_path)
                print(f"🚀 Shipping submitted — sending order {order_id} to printer")
                threading.Thread(
                    target=send_to_printer,
                    args=(interior_path, cover_path, user_id, order_id),
                    daemon=True,
                ).start()
            except Exception:
                print("⏸ Shipping saved, but cover PDF not ready/available yet.")
        else:
            print("⏸ Shipping saved, but PDFs not ready yet (will print after generation).")

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


@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        print("❌ Stripe webhook verification failed:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        stripe_session_id = session.get("id")
        meta = session.get("metadata", {}) or {}
        user_id = meta.get("user_id")
        upload_id = meta.get("upload_id")
        order_type = meta.get("order_type") or "pdf"
        title = meta.get("title") or "Bookify Order"

        if not user_id or not upload_id:
            print("⚠️ Missing user_id or upload_id in Stripe metadata")
            return JSONResponse(status_code=200, content={"status": "missing fields"})

        status_order = "Complete" if order_type == "pdf" else "Processing"

        order_insert = supabase.table("orders").insert({
            "user_id": user_id,
            "title": title,
            "type": order_type,
            "status": status_order,
            "stripe_session_id": stripe_session_id,
        }).execute()

        order_id = order_insert.data[0]["id"] if order_insert.data else None
        print(f"✅ Payment confirmed for upload {upload_id} (user {user_id}) → order {order_id}")

        upload_row = (
            supabase.table("upload_sessions")
            .select("audio_path")
            .eq("id", upload_id)
            .single()
            .execute()
        )
        audio_path = upload_row.data["audio_path"] if upload_row.data else None
        if not audio_path:
            print(f"⚠️ No audio_path found in upload_sessions for {upload_id}")
            return JSONResponse(status_code=200, content={"status": "missing audio_path"})

        print(f"🔁 Reprocessing {upload_id} from Supabase storage: {audio_path}")

        try:
            file_data = supabase.storage.from_("raw_audio").download(audio_path)
            # Preserve extension if possible
            suffix = os.path.splitext(audio_path)[1] or ".bin"
            temp_path = tempfile.mktemp(suffix=suffix)

            with open(temp_path, "wb") as f:
                f.write(getattr(file_data, "content", file_data))

            print(f"✅ Downloaded raw audio to {temp_path}")

            threading.Thread(
                target=process_audio,
                args=(upload_id, temp_path, user_id, True, order_id),
                daemon=True,
            ).start()

            supabase.table("upload_sessions").update({"status": "processing"}).eq("id", upload_id).execute()

        except Exception as e:
            print(f"❌ Error reprocessing {upload_id} from Supabase:", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    return JSONResponse(status_code=200, content={"status": "success"})


# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
