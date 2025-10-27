from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os, tempfile, subprocess, json
import imageio_ffmpeg as ffmpeg
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# --- CORS ---
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

# --- get ffmpeg binary path ---
FFMPEG = ffmpeg.get_ffmpeg_exe()

# --- helper: compress audio ---
def compress_audio(input_path: str) -> str:
    """Convert to 64 kbps mono MP3 to reduce size for Whisper."""
    output_path = tempfile.mktemp(suffix=".mp3")
    subprocess.run(
        [FFMPEG, "-y", "-i", input_path, "-ac", "1", "-b:a", "64k", output_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return output_path

# --- helper: get duration via ffmpeg ---
def get_duration(file_path: str) -> float:
    """Return duration (seconds) using ffmpeg -show_entries json."""
    cmd = [
        FFMPEG, "-i", file_path,
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            # Example: Duration: 00:02:12.34,
            parts = line.strip().split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0

# --- helper: split large file into ~10 MB chunks ---
def chunk_audio(file_path: str, chunk_size_mb=10) -> list[str]:
    size = os.path.getsize(file_path)
    if size <= chunk_size_mb * 1024 * 1024:
        return [file_path]

    dur = get_duration(file_path)
    if dur <= 0:
        return [file_path]

    n_parts = int(size // (chunk_size_mb * 1024 * 1024)) + 1
    part_dur = dur / n_parts
    chunk_paths = []

    for i in range(n_parts):
        part_path = tempfile.mktemp(suffix=f"_part{i}.mp3")
        start_time = i * part_dur
        subprocess.run(
            [FFMPEG, "-y", "-i", file_path,
             "-ss", str(start_time), "-t", str(part_dur),
             "-acodec", "copy", part_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        chunk_paths.append(part_path)

    return chunk_paths

# --- main upload route ---
@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    compressed_path = compress_audio(temp_path)
    chunks = chunk_audio(compressed_path)
    transcripts = []

    for idx, path in enumerate(chunks, start=1):
        print(f"🎧 Processing chunk {idx}/{len(chunks)}: {path}")
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

    # Summarize into chapters
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an editor who turns long-form spoken transcripts into "
                    "structured, chaptered summaries. Each chapter should focus on a major topic shift, "
                    "speaker transition, or narrative milestone. Each chapter must include a clear title "
                    "and 2–5 short paragraphs summarizing that section. Do not invent new details; "
                    "only rephrase or clarify what was said."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Divide this transcript into chapters based on topic or scene changes. "
                    f"Return each chapter with a clear title and narrative flow:\n\n{full_text}"
                ),
            },
        ],
        max_tokens=4000,
    )

    return {"chapters": completion.choices[0].message.content}

# optional: serve frontend if bundled
if os.path.exists("static/dist"):
    app.mount("/", StaticFiles(directory="static/dist", html=True), name="static")
