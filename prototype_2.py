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


# --- background job that saves to results ---
def process_audio(upload_id: str, temp_path: str, user_id: str, has_paid: bool):
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

        # --- 🧠 GPT-4o processing (your original system prompt preserved)
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
                        "that capture the key moments, transitions, and emotions of that section."
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

        # --- ✂️ PREVIEW MODE (if unpaid, show partial output)
        if not has_paid:
            preview_lines = chapters_text.splitlines()[:40]  # first ~40 lines
            preview_text = "\n".join(preview_lines) + "\n\n[...] Unlock full text with payment."
            results[upload_id] = {
                "status": "done",
                "chapters": preview_text,
                "is_preview": True,
            }
            print(f"User {user_id} received preview only.")
            return

        # --- 💾 Save full content for paying users
        results[upload_id] = {
            "status": "done",
            "chapters": chapters_text,
            "is_preview": False,
        }

        try:
            supabase.table("user_books").insert({
                "user_id": user_id,
                "title": f"Session {upload_id[:8]}",
                "content": chapters_text,
            }).execute()
            print(f"Saved full book for user {user_id}.")
        except Exception as db_err:
            print("Supabase insert failed:", db_err)

    except Exception as e:
        results[upload_id] = {"status": "error", "error": str(e)}
        print("Error in process_audio:", e)


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

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), authorization: str = Header(None)):
    user_id = verify_token(authorization)

    # Try to get existing profile row
    user_resp = supabase.table("profiles").select("has_paid").eq("id", user_id).execute()
    user_data = user_resp.data or []

    if not user_data:
        print(f"⚠️ No profile found for {user_id}, creating one...")
        supabase.table("profiles").insert({
            "id": user_id,
            "has_paid": False
        }).execute()
        has_paid = False
    else:
        has_paid = user_data[0].get("has_paid", False)

    upload_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    results[upload_id] = {"status": "processing"}
    threading.Thread(
        target=process_audio,
        args=(upload_id, temp_path, user_id, has_paid),
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

    if purchase_type == "pdf":
        price_id = os.getenv("STRIPE_PRICE_PDF")
        success_url = "https://speech-to-text-o5lh.onrender.com/pdf-download"
    elif purchase_type == "book":
        price_id = os.getenv("STRIPE_PRICE_BOOK")
        success_url = "https://speech-to-text-o5lh.onrender.com/book-customize"
    else:
        return JSONResponse({"error": "Invalid type"}, status_code=400)

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="payment",
        customer_email=None,  # You could populate via Supabase user email if desired
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url="https://speech-to-text-o5lh.onrender.com/upload",
        metadata={"user_id": user_id, "purchase_type": purchase_type},
    )
    return {"url": session.url}


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
        user_id = session["metadata"]["user_id"]

        print("✅ Stripe webhook received for user:", user_id)
        supabase.table("profiles").update({"has_paid": True}).eq("id", user_id).execute()

    return JSONResponse(status_code=200, content={"status": "success"})


# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
