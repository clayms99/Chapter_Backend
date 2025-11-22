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
import requests


load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
LULU_CLIENT_KEY = os.getenv("LULU_CLIENT_KEY")
LULU_CLIENT_SECRET = os.getenv("LULU_CLIENT_SECRET")
LULU_BASE_URL = os.getenv("LULU_BASE_URL", "https://api.sandbox.lulu.com")
LULU_POD_PACKAGE_ID = os.getenv("LULU_POD_PACKAGE_ID")
LULU_CONTACT_EMAIL = os.getenv("LULU_CONTACT_EMAIL", "you@yourdomain.com")

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

def get_lulu_token() -> str:
    """
    Get OAuth2 access token from Lulu using client_credentials.
    """
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

def send_to_printer(storage_path: str, user_id: str, order_id: str):
    """
    storage_path: path in 'book_files' bucket (e.g. 'books/<user>/<upload>.pdf')
    """

    # 1) Get public URL to the PDF
    public_res = supabase.storage.from_("book_files").get_public_url(storage_path)

    if isinstance(public_res, str):
        public_url = public_res
    else:
        # Fallback if a future version returns a dict-like structure
        public_url = (
            getattr(public_res, "public_url", None)
            or getattr(public_res, "publicUrl", None)
            or (public_res.get("publicUrl") if isinstance(public_res, dict) else None)
            or (public_res.get("public_url") if isinstance(public_res, dict) else None)
        )

    if not public_url:
        print(f"❌ Could not get public URL for {storage_path}: {public_res}")
        return

    print(f"🌐 Lulu will fetch interior PDF from {public_url}")

    # 2) (Optional for now) – load shipping address from your DB
    #    For MVP you could ship everything to yourself or a fixed address.
    #    Later, you'll store shipping info on `orders` from BookCustomize.tsx.
    order_row = (
        supabase.table("orders")
        .select("*")
        .eq("id", order_id)
        .single()
        .execute()
    )
    order_data = order_row.data or {}

    # TODO: replace these with real columns once BookCustomize collects them
    shipping_address = {
    "name": order_data.get("ship_name", "Bookify Test User"),
    "street1": order_data.get("ship_line1", "123 Test St"),
    "city": order_data.get("ship_city", "Durham"),
    "state_code": order_data.get("ship_state", "NC"),
    "country_code": order_data.get("ship_country", "US"),
    "postcode": order_data.get("ship_postal", "27701"),
    "phone_number": order_data.get("ship_phone", "+15555555555"),
    }


    # 3) Build the Lulu Print-Job payload
    #    See Lulu docs: POST /print-jobs/ with line_items, shipping_address,
    #    shipping_option_level, contact_email, etc. :contentReference[oaicite:3]{index=3}

    payload = {
        "contact_email": LULU_CONTACT_EMAIL,
        "shipping_address": shipping_address,
        "shipping_option_level": "MAIL",   # or PRIORITY_MAIL, GROUND, EXPEDITED, EXPRESS
        "external_id": str(order_id),

        "line_items": [
            {
                "pod_package_id": LULU_POD_PACKAGE_ID,
                "quantity": 1,
                "title": order_data.get("title", f"Bookify Order {order_id[:8]}"),
                "printable_normalization": {
                    # In a perfect world you'd provide separate interior + cover PDFs.
                    # For MVP you can use the same PDF for both, or just interior
                    # if your product/package allows it.
                    "cover": {
                        "source_url": public_url,
                    },
                    "interior": {
                        "source_url": public_url,
                    },
                },
            }
        ],
    }

    token = get_lulu_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("📦 Creating Lulu Print-Job...")
    resp = requests.post(
        f"{LULU_BASE_URL}/print-jobs/",
        json=payload,
        headers=headers,
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"❌ Lulu print-job error {resp.status_code}: {resp.text}")
        # optionally update orders.status = 'Print Error'
        supabase.table("orders").update({"status": "Print Error"}).eq("id", order_id).execute()
        return

    job = resp.json()
    print(f"✅ Lulu Print-Job created: {job.get('id')}")
    # You may want to save Lulu's job ID:
    supabase.table("orders").update({
        "status": "Printing",
        "lulu_job_id": job.get("id")
    }).eq("id", order_id).execute()




# --- background job that saves to results ---
def process_audio(upload_id: str, temp_path: str, user_id: str, has_paid: bool, order_id: str | None = None):
    print(f"▶️ process_audio START upload_id={upload_id}, has_paid={has_paid}, order_id={order_id}, temp_path={temp_path}")

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

        # ❗ only delete source files *after* successful paid processing
        if has_paid:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
        else:
            print(f"⏸ Keeping temp file for preview {upload_id} to allow reprocess later.")

        # --- GPT-4o and rest unchanged ---


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
        print(f"✅ Whisper + GPT done for upload_id={upload_id}, has_paid={has_paid}")
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
        # --- 🔗 Link book to order if it exists ---
        if order_id and book_id:
            supabase.table("orders").update({"book_id": book_id}).eq("id", order_id).execute()
            print(f"✅ Linked book {book_id} → order {order_id}")

            # If this order was a “book” purchase, send to printer
            order_data = supabase.table("orders").select("type").eq("id", order_id).single().execute()
            if order_data.data and order_data.data["type"] == "book":
                print("🚀 Sending book to printer API...")
                # Pass storage path (in your bucket), not local /tmp path
                send_to_printer(storage_path, user_id, order_id)


        # --- 🏁 Mark upload as fully complete in DB ---
        try:
            supabase.table("upload_sessions").update({
                "status": "complete"
            }).eq("id", upload_id).execute()
            print(f"🏁 upload_sessions status updated to 'complete' for {upload_id}")
        except Exception as e:
            print(f"⚠️ Failed to update upload_sessions status for {upload_id}: {e}")


    except Exception as e:
        import traceback
        traceback.print_exc()
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

    # Save the raw audio to Supabase storage
    storage_audio_path = f"uploads/{user_id}/{upload_id}.mp3"
    with open(temp_path, "rb") as f:
        supabase.storage.from_("raw_audio").upload(storage_audio_path, f)
    print(f"✅ Uploaded raw audio to Supabase: {storage_audio_path}")

    # Save the upload metadata in Supabase for webhook access
    supabase.table("upload_sessions").insert({
        "id": upload_id,
        "user_id": user_id,
        "audio_path": storage_audio_path,
        "status": "preview",
    }).execute()
    print(f"🗂️ Recorded upload session {upload_id} in Supabase")

    results[upload_id] = {"status": "processing", "paid": False}

    threading.Thread(
        target=process_audio,
        args=(upload_id, temp_path, user_id, False),
        daemon=False,
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
    """Handles Stripe webhook events to confirm payment and trigger full book generation."""
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
            print("⚠️ Missing user_id or upload_id in Stripe metadata")
            return JSONResponse(status_code=200, content={"status": "missing fields"})

        # ✅ Create an order record
        status_order = "Complete" if order_type == "pdf" else "Processing"
        order_insert = supabase.table("orders").insert({
            "user_id": user_id,
            "title": title,
            "type": order_type,
            "status": status_order
        }).execute()

        order_id = order_insert.data[0]["id"] if order_insert.data else None
        print(f"✅ Payment confirmed for upload {upload_id} (user {user_id}) → order {order_id}")

        # ✅ Look up stored audio path from upload_sessions
        upload_row = (
            supabase.table("upload_sessions")
            .select("audio_path")
            .eq("id", upload_id)
            .single()
            .execute()
        )
        audio_path = upload_row.data["audio_path"] if upload_row.data else None

        if not audio_path:
            print(f"⚠️ No audio_path found in upload_sessions for {upload_id}, cannot reprocess.")
            return JSONResponse(status_code=200, content={"status": "missing audio_path"})

        print(f"🔁 Reprocessing {upload_id} from Supabase storage: {audio_path}")

        try:
            # Download the audio file from Supabase Storage
            file_data = supabase.storage.from_("raw_audio").download(audio_path)
            temp_path = tempfile.mktemp(suffix=".mp3")
            with open(temp_path, "wb") as f:
                f.write(getattr(file_data, "content", file_data))
            print(f"✅ Downloaded raw audio to {temp_path}")

            # Launch background thread to generate full book and PDF
            threading.Thread(
                target=process_audio,
                args=(upload_id, temp_path, user_id, True, order_id),
                daemon=False,
            ).start()

            # Update upload_sessions to mark completion
            supabase.table("upload_sessions").update({"status": "processing"}).eq("id", upload_id).execute()
        except Exception as e:
            print(f"❌ Error reprocessing {upload_id} from Supabase:", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    # Respond to Stripe immediately (do not wait for background thread)
    return JSONResponse(status_code=200, content={"status": "success"})


# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
