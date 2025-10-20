from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os, tempfile
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow the frontend’s Render URL (we’ll add this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",                      # local dev
        "https://speech-to-text-o5lh.onrender.com",        # your frontend Render URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
        temp_audio.write(await file.read())
        temp_path = temp_audio.name

    with open(temp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    text = transcript.text

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Convert conversations into short chapter-style narratives."},
            {"role": "user", "content": f"Turn this into a short chapter:\n\n{text}"}
        ],
        max_tokens=2000
    )
    os.remove(temp_path)
    return {"chapters": completion.choices[0].message.content}
