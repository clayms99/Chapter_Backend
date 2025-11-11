from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os, tempfile, subprocess, threading, uuid
import imageio_ffmpeg as ffmpeg
from dotenv import load_dotenv
import stripe
from supabase import create_client, Client
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Header
import jwt  # from PyJWT, not jose
from fastapi import Header, HTTPException, status
from fastapi.responses import StreamingResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from textwrap import wrap
from reportlab.lib.units import inch
from fastapi import Depends
from fastapi import Form
from fastapi.responses import Response

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
import tempfile


stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

# --- store results in memory ---
results = {}

# --- helpers (same as before) ---
def compress_audio(input_path: str) -> str:
    output_path = tempfile.mktemp(suffix=".mp3")
    subprocess.run(
        [FFMPEG, "-y", "-i", input_path, "-ac", "1", "-b:a", "64k", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return output_path


def chunk_audio(file_path: str, chunk_size_mb=10) -> list[str]:
    size = os.path.getsize(file_path)
    if size <= chunk_size_mb * 1024 * 1024:
        return [file_path]

    # Use ffprobe for accurate duration
    dur = float(
        subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]).decode().strip()
    )

    n_parts = int(size // (chunk_size_mb * 1024 * 1024)) + 1
    part_dur = dur / n_parts
    chunk_paths = []

    for i in range(n_parts):
        part_path = tempfile.mktemp(suffix=f"_part{i}.mp3")
        start_time = i * part_dur
        subprocess.run(
            [FFMPEG, "-y", "-i", file_path, "-ss", str(start_time), "-t", str(part_dur),
             "-acodec", "copy", part_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        chunk_paths.append(part_path)

    return chunk_paths

def make_book_pdf(chapter_text: str, user_name: str = "User") -> str:
    """Generate a clean multi-page PDF from the chapter text."""
    styles = getSampleStyleSheet()
    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=LETTER,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    story = []
    story.append(Paragraph(f"<b>{user_name}'s Book</b>", styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))

    for paragraph in chapter_text.split("\n\n"):
        story.append(Paragraph(paragraph.strip(), styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    return pdf_path

def send_to_printer(pdf_path: str, user_id: str, order_id: str):
    print(f"🖨️ (stub) Would send {pdf_path} to print for user {user_id}, order {order_id}")
    # TODO: Replace this with Lulu API integration later
    # Example:
    # response = requests.post("https://api.lulu.com/print-jobs/", ...)
    # supabase.table("orders").update({"status": "Printing"}).eq("id", order_id).execute()



# --- background job that saves to results ---
def process_audio(upload_id: str, temp_path: str, user_id: str, has_paid: bool, order_id: str | None = None):
    try:
        compressed_path = compress_audio(temp_path)
        chunks = chunk_audio(compressed_path)
        transcripts = []

        for i, path in enumerate(chunks):
            with open(path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            transcripts.append(transcript.text)
            os.remove(path)

        full_text = "\n".join(transcripts)
        os.remove(temp_path)
        if os.path.exists(compressed_path):
            os.remove(compressed_path)

        # --- 🧠 GPT-4o processing ---
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
                {
                    "role": "user",
                    "content": f"Divide this transcript into chapters:\n\n{full_text}",
                },
            ],
            max_tokens=4000,
        )

        chapters_text = completion.choices[0].message.content.strip()

        # --- ✂️ PREVIEW MODE ---
        if not has_paid:
            preview_lines = chapters_text.splitlines()[:20]
            preview_text = "\n".join(preview_lines) + "\n\n[...] Unlock full text with payment."
            results[upload_id] = {
                "status": "done",
                "chapters": preview_text,
                "is_preview": True,
            }
            print(f"User {user_id} received preview only.")
            return

        # --- 🧾 Paid version ---
        results[upload_id] = {
            "status": "done",
            "chapters": chapters_text,
            "is_preview": False,
        }

        # --- 🖨️ Generate PDF for printing ---
        pdf_path = make_book_pdf(chapters_text, user_id)
        print(f"✅ Created book PDF at {pdf_path}")

        # --- 💾 Upload to Supabase Storage ---
        storage_path = f"books/{user_id}/{upload_id}.pdf"
        with open(pdf_path, "rb") as f:
            supabase.storage.from_("book_files").upload(storage_path, f)
        print(f"✅ Uploaded PDF to Supabase storage: {storage_path}")

        # --- 🧠 Insert into user_books ---
        book_insert = supabase.table("user_books").insert({
            "user_id": user_id,
            "title": f"Session {upload_id[:8]}",
            "content": chapters_text,
            "pdf_path": storage_path,   # 👈 optional column to store where the file lives
        }).execute()

        book_id = None
        if book_insert.data:
            book_id = book_insert.data[0]["id"]
            print(f"✅ Saved book {book_id} for user {user_id}")

        # --- 🔗 Link book to order if it exists ---
        if order_id and book_id:
            supabase.table("orders").update({"book_id": book_id}).eq("id", order_id).execute()
            print(f"✅ Linked book {book_id} → order {order_id}")

            # If this order was a “book” purchase, send to printer
            order_data = supabase.table("orders").select("type").eq("id", order_id).single().execute()
            if order_data.data and order_data.data["type"] == "book":
                print("🚀 Sending book to printer API...")
                send_to_printer(pdf_path, user_id, order_id)

    except Exception as e:
        results[upload_id] = {"status": "error", "error": str(e)}
        print("❌ Error in process_audio:", e)


def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}  # 👈 disable audience check
        )
        print("✅ TOKEN PAYLOAD:", payload)
        return payload.get("sub")
    except Exception as e:
        print("❌ JWT decode failed:", str(e))
        raise HTTPException(status_code=401, detail="Invalid token")

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(status_code=200)

@app.post("/upload/")
async def upload_audio(
    file: UploadFile = File(...),
    authorization: str = Header(None),
):
    user_id = verify_token(authorization)
    upload_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    results[upload_id] = {
    "status": "processing",
    "paid": False,
    "temp_path": temp_path,  # 👈 add this
    }


    threading.Thread(
        target=process_audio,
        args=(upload_id, temp_path, user_id, False),  # 👈 always unpaid initially
        daemon=True,
    ).start()

    return {"id": upload_id, "status": "processing"}


@app.get("/result/{upload_id}")
def get_result(upload_id: str):
    return results.get(upload_id, {"status": "not_found"})

DOMAIN = "https://speech-to-text-o5lh.onrender.com"  # your frontend URL

@app.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    data = await request.json()
    purchase_type = data.get("type")
    user_id = data.get("user_id")
    upload_id = data.get("upload_id")  # 👈 new

    if purchase_type == "pdf":
        price_id = os.getenv("STRIPE_PRICE_PDF")
        success_url = f"{DOMAIN}/pdf-download"
    elif purchase_type == "book":
        price_id = os.getenv("STRIPE_PRICE_BOOK")
        success_url = f"{DOMAIN}/book-customize"
    else:
        return JSONResponse({"error": "Invalid type"}, status_code=400)

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="payment",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=f"{DOMAIN}/upload",
        metadata={
            "user_id": user_id,
            "upload_id": upload_id,  # 👈 pass this through
            "order_type": purchase_type,
            "title": "Bookify Order",
        },
    )

    return {"url": session.url}


@app.get("/download-latest-pdf")
async def download_latest_pdf(authorization: str = Header(None)):
    try:
        user_id = verify_token(authorization)
        print(f"✅ Authenticated PDF download for user: {user_id}")

        res = (
            supabase.table("user_books")
            .select("content")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="No book found for this user.")

        book_text = res.data[0]["content"]

        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Text object lets you handle multi-line wrapping and spacing
        textobject = p.beginText()
        textobject.setTextOrigin(inch, height - inch)
        textobject.setFont("Helvetica", 12)
        line_height = 14
        max_width = width - 2 * inch

        for paragraph in book_text.split("\n"):
            # wrap each paragraph to fit within the page width (~90 chars)
            wrapped_lines = wrap(paragraph, 90)
            for line in wrapped_lines:
                textobject.textLine(line)
            textobject.textLine("")  # blank line between paragraphs

            # Handle page overflow
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

    except HTTPException as e:
        raise e
    except Exception as e:
        print("❌ Error generating PDF:", e)
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {e}")
    
@app.get("/download/{order_id}")
async def download_order_pdf(order_id: str, authorization: str = Header(None)):
    user_id = verify_token(authorization)
    print(f"✅ Authenticated PDF download for user {user_id}, order {order_id}")

    # Try to get the book_id linked to this order
    order_res = (
        supabase.table("orders")
        .select("book_id")
        .eq("id", order_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    book_id = None
    if order_res.data and order_res.data[0].get("book_id"):
        book_id = order_res.data[0]["book_id"]
        print(f"✅ Found linked book {book_id} for order {order_id}")
    else:
        print("⚠️ No linked book_id — falling back to latest user book")
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

    # Fetch the actual book content
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

    # Generate PDF
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


@app.get("/orders/{user_id}")
async def get_orders(user_id: str):
    res = supabase.table("orders").select("*").eq("user_id", user_id).execute()
    return res.data

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
        meta = session.get("metadata", {})
        user_id = meta.get("user_id")
        upload_id = meta.get("upload_id")
        order_type = meta.get("order_type") or "pdf"
        title = meta.get("title") or "Bookify Order"

        if not user_id or not upload_id:
            print("⚠️ Missing user_id or upload_id")
            return JSONResponse(status_code=200, content={"status": "missing fields"})

        # ✅ Create order record
        status_order = "Complete" if order_type == "pdf" else "Processing"
        order_insert = supabase.table("orders").insert({
            "user_id": user_id,
            "title": title,
            "type": order_type,
            "status": status_order
        }).execute()
        order_id = order_insert.data[0]["id"] if order_insert.data else None

        # ✅ Mark upload paid
        upload_state = results.get(upload_id, {})
        results[upload_id] = {**upload_state, "paid": True, "order_id": order_id}
        print(f"✅ Payment confirmed for upload {upload_id} (user {user_id})")

        # 🔁 If it already finished in preview mode, rerun as paid
        if upload_state.get("status") == "done" and upload_state.get("is_preview"):
            temp_path = upload_state.get("temp_path")
            if temp_path and os.path.exists(temp_path):
                print(f"🔁 Reprocessing {upload_id} now that payment is confirmed...")
                threading.Thread(
                    target=process_audio,
                    args=(upload_id, temp_path, user_id, True, order_id),
                    daemon=True,
                ).start()
            else:
                print(f"⚠️ No temp file left for {upload_id}, cannot reprocess.")

    return JSONResponse(status_code=200, content={"status": "success"})


# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
