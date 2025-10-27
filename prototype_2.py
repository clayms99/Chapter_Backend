from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os, tempfile, subprocess, threading, uuid
import imageio_ffmpeg as ffmpeg
from dotenv import load_dotenv

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
def process_audio(upload_id: str, temp_path: str):
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

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an editor who turns long-form spoken transcripts into "
                        "structured, chaptered summaries. Each chapter should focus on a major topic shift, "
                        "speaker transition, or narrative milestone. Each chapter must include a clear title "
                        "and 2–5 short paragraphs summarizing that section."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Divide this transcript into chapters:\n\n{full_text}",
                },
            ],
            max_tokens=4000,
        )

        results[upload_id] = {
            "status": "done",
            "chapters": completion.choices[0].message.content,
        }

    except Exception as e:
        results[upload_id] = {"status": "error", "error": str(e)}


@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    results[upload_id] = {"status": "processing"}
    threading.Thread(target=process_audio, args=(upload_id, temp_path), daemon=True).start()
    return {"id": upload_id, "status": "processing"}


@app.get("/result/{upload_id}")
def get_result(upload_id: str):
    return results.get(upload_id, {"status": "not_found"})


# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
