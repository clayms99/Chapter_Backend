# Booksly — Backend

Audio-to-book conversion and print-on-demand API. Users record a conversation, and the backend transcribes it, generates structured chapters, produces print-ready PDFs, and submits print jobs.

## Flow Diagram

The full application flow is documented in [`flow_diagram.mmd`](flow_diagram.mmd) (Mermaid format). It covers:

- **📤 Audio Upload** — upload, transcribe, preview
- **💳 Stripe Payment** — checkout, webhook, paid book generation
- **📬 Shipping & Printing** — address collection, Lulu print-on-demand
- **🔍 Query Endpoints** — order lookups and status polling
- **📥 Download Endpoints** — PDF retrieval

> **Tip:** Render the diagram with the [Mermaid Live Editor](https://mermaid.live), the GitHub Mermaid preview, or a VS Code Mermaid extension.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Database & Storage | [Supabase](https://supabase.com/) (PostgreSQL + object storage) |
| AI / Transcription | [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text) + GPT-4o-mini |
| PDF Generation | [ReportLab](https://www.reportlab.com/) |
| Payments | [Stripe](https://stripe.com/) (Checkout + Webhooks) |
| Print-on-Demand | [Lulu API](https://developers.lulu.com/) |
| Audio Processing | FFmpeg via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| Auth | Supabase JWT (verified with [PyJWT](https://pyjwt.readthedocs.io/)) |

## Project Structure

```
Chapter_Backend/
├── prototype_2.py        # Main FastAPI application (all routes, services, helpers)
├── flow_diagram.mmd      # Mermaid flow diagram of the backend
├── requirements.txt      # Python dependencies
├── render-build.sh       # Build script for Render deployment
└── .gitignore
```

### Code Layout (`prototype_2.py`)

The application is organized into logical sections within a single file:

| Section | Description |
|---------|------------|
| **Environment & Clients** | Loads env vars; initialises Supabase, Stripe, and OpenAI clients |
| **Auth** | `verify_token()` — validates Supabase JWTs from the `Authorization` header |
| **Font Management** | Registers TTF fonts (bundled or system fallback) for ReportLab |
| **Audio Processing** | `compress_to_mp3()`, `get_duration_seconds()`, `chunk_audio_mp3()` — normalize, measure, and split audio |
| **PDF Generation** | `make_interior_pdf()`, `make_cover_pdf()` — Lulu-compliant interior and cover PDFs |
| **Lulu & Supabase Helpers** | `get_lulu_token()`, `supabase_public_url()`, `send_to_printer()` |
| **Background Processing** | `process_audio()` — orchestrates transcription, chapter generation, PDF creation, and storage |
| **API Routes** | All FastAPI endpoints (see below) |

## API Endpoints

### Audio Upload

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/upload/` | Bearer token | Upload audio file; returns `upload_id` and starts background processing |
| `GET` | `/result/{upload_id}` | — | Poll processing status and retrieve the preview or final result |

### Payments

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/create-checkout-session` | — | Create a Stripe Checkout session for a PDF or book purchase |
| `POST` | `/webhook` | Stripe signature | Stripe webhook — processes `checkout.session.completed` events |

### Orders & Shipping

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/orders/{user_id}` | — | List all orders for a user |
| `GET` | `/order-from-session` | Bearer token | Look up an order by Stripe session ID |
| `POST` | `/submit-shipping` | Bearer token | Submit shipping address and custom title; triggers printing if the book is ready |

### Downloads

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/download-latest-pdf` | Bearer token | Download the user's most recent book PDF |
| `GET` | `/download/{order_id}` | Bearer token | Download the PDF for a specific order |

## Database Schema (Supabase)

### Tables

| Table | Key Columns | Purpose |
|-------|------------|---------|
| `upload_sessions` | `id`, `user_id`, `audio_path`, `status` | Tracks audio uploads and processing state |
| `orders` | `id`, `user_id`, `type`, `status`, `stripe_session_id`, shipping fields, `book_id`, `lulu_job_id` | Purchase and fulfilment records |
| `user_books` | `id`, `user_id`, `title`, `content`, `pdf_path`, `created_at` | Generated books with chapter content and PDF storage path |

### Storage Buckets

| Bucket | Path Pattern | Contents |
|--------|-------------|----------|
| `raw_audio` | `uploads/{user_id}/{upload_id}.{ext}` | Original audio files |
| `book_files` | `books/{user_id}/{upload_id}.pdf` | Interior and cover PDFs |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key (Whisper + GPT) |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service-role key |
| `SUPABASE_JWT_SECRET` | Yes | Secret used to verify Supabase JWTs |
| `STRIPE_SECRET_KEY` | Yes | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | Yes | Stripe webhook signing secret |
| `STRIPE_PRICE_PDF` | Yes | Stripe Price ID for PDF purchases |
| `STRIPE_PRICE_BOOK` | Yes | Stripe Price ID for book purchases |
| `LULU_CLIENT_KEY` | No | Lulu API client key |
| `LULU_CLIENT_SECRET` | No | Lulu API client secret |
| `LULU_BASE_URL` | No | Lulu API base URL (defaults to sandbox) |
| `LULU_POD_PACKAGE_ID` | No | Lulu print-on-demand package ID |
| `LULU_CONTACT_EMAIL` | No | Contact email for Lulu print jobs |

## Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg (installed automatically by `imageio-ffmpeg`)

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
cp .env.example .env   # fill in the required variables
uvicorn prototype_2:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs are served at `/docs` (Swagger UI).

## Deployment

The project includes `render-build.sh` for deploying on [Render](https://render.com/):

```bash
#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install -r requirements.txt
pip install imageio-ffmpeg
```

Set the **Start Command** to:

```
uvicorn prototype_2:app --host 0.0.0.0 --port $PORT
```

Configure all required environment variables in the Render dashboard.
