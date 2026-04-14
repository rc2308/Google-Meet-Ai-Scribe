import os
import time
import json
import asyncio
import httpx
import boto3
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ──────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

RECALL_BASE = f"https://{os.getenv('RECALL_REGION', 'us-east-1')}.recall.ai/api/v1"
RECALL_HEADERS = {"Authorization": f"Token {os.getenv('RECALL_API_KEY')}"}

HF_API_KEY  = os.getenv("HF_API_KEY")
HF_PROVIDER = os.getenv("HF_PROVIDER", "fireworks-ai")

# ── Sarvam ───────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE    = "https://api.sarvam.ai"
SARVAM_HEADERS = {"api-subscription-key": SARVAM_API_KEY}

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
}

# ── Status map ───────────────────────────────────────────
TERMINAL_OK = {"done", "call_ended", "recording_done", "analysis_done", "media_expired"}
TERMINAL_FAIL = {"fatal", "error", "recording_permission_denied", "bot_kicked", "rejected"}

def map_status(status: str) -> dict:
    mapping = {
        "ready":                      ("joining",    "🤖 Bot is ready — attempting to join the call…"),
        "joining_call":               ("joining",    "🔗 Bot is joining the meeting…"),
        "in_waiting_room":            ("joining",    "⏳ Bot is in the waiting room — please admit it!"),
        "in_call_not_recording":      ("recording",  "📞 Bot joined the call — starting recording…"),
        "recording_permission_allowed": ("recording","✅ Recording permission granted!"),
        "recording_permission_denied":  ("recording","❌ Recording permission denied by host."),
        "in_call_recording":          ("recording",  "🔴 Bot is actively recording the meeting…"),
        "recording_done":             ("processing", "💾 Recording finished — processing audio…"),
        "call_ended":                 ("processing", "📵 Call ended — waiting for recording upload…"),
        "done":                       ("processing", "✅ Recall processing complete. Fetching audio…"),
        "analysis_done":              ("processing", "🔍 Analysis done. Fetching audio…"),
    }
    step, msg = mapping.get(status, ("joining", f"ℹ️ Bot status: {status}"))
    return {"step": step, "msg": msg}

# ── SSE helper ───────────────────────────────────────────
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ── Recall helpers ───────────────────────────────────────
async def create_bot(meet_link: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"{RECALL_BASE}/bot/",
            json={"meeting_url": meet_link, "bot_name": "AI Scribe"},
            headers=RECALL_HEADERS,
        )
        res.raise_for_status()
        return res.json()

async def get_bot(bot_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.get(f"{RECALL_BASE}/bot/{bot_id}/", headers=RECALL_HEADERS)
        res.raise_for_status()
        return res.json()

# ── Audio conversion helper ───────────────────────────────
def _get_ffmpeg_exe() -> str:
    """Return path to ffmpeg binary (imageio-ffmpeg if available, else system)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to a 16 kHz mono WAV that Whisper loves.
    Returns the path to the new WAV file (caller must delete it).
    """
    import subprocess
    ffmpeg_exe = _get_ffmpeg_exe()
    wav_path = input_path.replace(Path(input_path).suffix, "_16k.wav")
    proc = subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-ar", "16000",   # 16 kHz sample rate — Whisper's native rate
            "-ac", "1",       # mono
            "-c:a", "pcm_s16le",
            wav_path,
        ],
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        # Conversion failed — log and return original path as fallback
        print(f"[ffmpeg convert] warning: {proc.stderr.decode()[:300]}")
        return input_path
    print(f"[ffmpeg convert] → {wav_path}")
    return wav_path

# ── Whisper via Groq ─────────────────────────────────────
async def transcribe(audio_url: str) -> str:
    print(f"Downloading audio from: {audio_url}")

    is_presigned = "AWSAccessKeyId" in audio_url or "X-Amz-Signature" in audio_url
    headers = {} if is_presigned else RECALL_HEADERS

    async with httpx.AsyncClient(timeout=300) as client:
        res = await client.get(audio_url, headers=headers, follow_redirects=True)
        res.raise_for_status()
        audio_bytes = res.content

    size_mb = len(audio_bytes) / 1024 / 1024
    print(f"Downloaded: {size_mb:.2f} MB")

    if len(audio_bytes) < 1000:
        raise ValueError(f"Download returned non-audio data: {audio_bytes[:200]}")

    # Write raw download to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(audio_bytes)
        raw_path = tmp.name

    # Convert to 16 kHz mono WAV for best Whisper accuracy
    wav_path = convert_to_wav(raw_path)
    converted = wav_path != raw_path   # True if ffmpeg succeeded

    try:
        print("Sending to Groq Whisper...")
        filename = "recording.wav" if converted else "recording.mp4"
        with open(wav_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, f),
                model="whisper-large-v3",
                response_format="text",
                prompt=(
                    "Transcript of a meeting or conversation. "
                    "Speakers may use Indian English, technical jargon, proper nouns such as "
                    "IIT, CGPA, NIT, ISRO, or mix English with Hindi or other Indian languages. "
                    "Transcribe every word exactly as spoken, preserving names and numbers accurately."
                ),
            )
        text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
        print(f"Transcript done — {len(text)} characters")
        return text
    finally:
        Path(raw_path).unlink(missing_ok=True)
        if converted:
            Path(wav_path).unlink(missing_ok=True)
        print("Temp file(s) deleted.")

# ── Sarvam STT — native language transcription (chunked for >30 s) ──
_CHUNK_SECS = 25   # Sarvam limit is 30 s; stay under with margin

async def _sarvam_send_chunk(client: httpx.AsyncClient, chunk_bytes: bytes,
                              language_code: str, label: str) -> str:
    """POST a single audio chunk to Sarvam and return its transcript."""
    res = await client.post(
        f"{SARVAM_BASE}/speech-to-text",
        files={"file": (f"{label}.mp4", chunk_bytes, "audio/mp4")},
        data={"model": "saarika:v2.5", "language_code": language_code},
        headers=SARVAM_HEADERS,
        timeout=120,
    )
    if res.status_code != 200:
        print(f"[Sarvam] chunk {label} error {res.status_code}: {res.text}")
        return ""
    text = res.json().get("transcript", "")
    print(f"[Sarvam] chunk {label}: {len(text)} chars")
    return text

async def _split_with_ffmpeg(tmp_path: str, chunk_dir: str) -> list[str]:
    """Split audio file into _CHUNK_SECS segments with 2 s overlap using ffmpeg."""
    import subprocess
    ffmpeg_exe = _get_ffmpeg_exe()

    pattern = os.path.join(chunk_dir, "chunk_%04d.mp4")
    proc = subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", tmp_path,
            "-f", "segment",
            "-segment_time", str(_CHUNK_SECS),
            "-segment_overlap", "2",
            "-c", "copy",
            "-reset_timestamps", "1",
            pattern,
        ],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[:400])
    chunks = sorted(
        os.path.join(chunk_dir, f)
        for f in os.listdir(chunk_dir)
        if f.startswith("chunk_")
    )
    return chunks

async def sarvam_transcribe(audio_url: str, language_code: str) -> str:
    """Download audio and transcribe with Sarvam saarika:v2.5 (Indian languages).
    Automatically chunks audio longer than 25 s to stay within the API limit.
    Falls back to Groq Whisper if chunking is unavailable.
    """
    print(f"[Sarvam] downloading audio for language={language_code}")
    is_presigned = "AWSAccessKeyId" in audio_url or "X-Amz-Signature" in audio_url
    dl_headers = {} if is_presigned else RECALL_HEADERS

    async with httpx.AsyncClient(timeout=300) as client:
        res = await client.get(audio_url, headers=dl_headers, follow_redirects=True)
        res.raise_for_status()
        audio_bytes = res.content

    size_mb = len(audio_bytes) / 1024 / 1024
    print(f"[Sarvam] downloaded {size_mb:.2f} MB")

    if len(audio_bytes) < 1000:
        raise ValueError(f"Non-audio data returned: {audio_bytes[:200]}")

    # Write to temp file for splitting
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    chunk_dir = tempfile.mkdtemp()
    try:
        # ── Try ffmpeg splitting ──────────────────────────────
        chunk_paths: list[str] = []
        try:
            chunk_paths = await _split_with_ffmpeg(tmp_path, chunk_dir)
            print(f"[Sarvam] ffmpeg split → {len(chunk_paths)} chunk(s)")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[Sarvam] ffmpeg unavailable ({e}); trying single-shot first")

        # ── If no chunks (no ffmpeg), attempt single-shot ────
        if not chunk_paths:
            async with httpx.AsyncClient(timeout=300) as client:
                res = await client.post(
                    f"{SARVAM_BASE}/speech-to-text",
                    files={"file": ("recording.mp4", audio_bytes, "audio/mp4")},
                    data={"model": "saarika:v2.5", "language_code": language_code},
                    headers=SARVAM_HEADERS,
                )
            if res.status_code == 200:
                text = res.json().get("transcript", "")
                print(f"[Sarvam] single-shot done — {len(text)} chars")
                return text
            elif "duration exceeds" in res.text or res.status_code == 400:
                print("[Sarvam] audio too long and ffmpeg unavailable — falling back to Groq Whisper")
                return await transcribe(audio_url)   # Groq Whisper fallback
            else:
                raise ValueError(f"Sarvam STT error {res.status_code}: {res.text}")

        # ── Transcribe each chunk in sequence ────────────────
        parts: list[str] = []
        async with httpx.AsyncClient(timeout=120) as client:
            for i, path in enumerate(chunk_paths):
                with open(path, "rb") as f:
                    chunk_bytes = f.read()
                text = await _sarvam_send_chunk(client, chunk_bytes, language_code, f"{i+1:03d}")
                if text:
                    parts.append(text)

        full = " ".join(parts)
        print(f"[Sarvam] chunked transcription complete — {len(full)} chars total")
        return full

    finally:
        Path(tmp_path).unlink(missing_ok=True)
        import shutil as _shutil
        _shutil.rmtree(chunk_dir, ignore_errors=True)


# ── Sarvam Translate — chunked (mayura:v1 max 1000 chars) ──────────
async def sarvam_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text via Sarvam mayura:v1. Chunks at 900 chars to stay under limit."""
    lines = text.split("\n")
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        line_len = len(line) + 1
        if cur_len + line_len > 900 and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [line], line_len
        else:
            cur.append(line)
            cur_len += line_len
    if cur:
        chunks.append("\n".join(cur))

    translated: list[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for chunk in chunks:
            if not chunk.strip():
                translated.append(chunk)
                continue
            res = await client.post(
                f"{SARVAM_BASE}/translate",
                json={
                    "input": chunk,
                    "source_language_code": source_lang,
                    "target_language_code": target_lang,
                    "model": "mayura:v1",
                    "mode": "formal",
                    "enable_preprocessing": False,
                },
                headers={**SARVAM_HEADERS, "Content-Type": "application/json"},
            )
            if res.status_code != 200:
                raise ValueError(f"Sarvam translate error {res.status_code}: {res.text}")
            translated.append(res.json().get("translated_text", chunk))

    return "\n".join(translated)

# ── gpt-oss-120b via HF ──────────────────────────────────
async def summarize(text: str) -> str:
    truncated = text[:12000] + "\n\n[transcript truncated for length]" if len(text) > 12000 else text
    print(f"Sending to gpt-oss-120b via HF (provider: {HF_PROVIDER})...")

    async with httpx.AsyncClient(timeout=180) as client:
        res = await client.post(
            "https://router.huggingface.co/v1/chat/completions",
            json={
                "model": f"openai/gpt-oss-120b:{HF_PROVIDER}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert meeting summarizer. You write accurate, human-friendly summaries that faithfully reflect what was actually said — nothing more, nothing less.",
                    },
                    {
                        "role": "user",
                        "content": f"""Summarize the following meeting transcript into structured notes.

OUTPUT FORMAT — you MUST follow this exactly:
- The VERY FIRST line must be: MEETING_TITLE: <a concise 3-7 word title that captures the topic, e.g. "Introduction — Raunak Chhatai" or "Q1 Budget Review" or "Team Standup — Sprint 12">
- After the title line, output the sections below
- Every section MUST start with its ## heading on its own line
- Each section body is a single flowing prose paragraph (NO bullet points, NO numbered lists)
- ALWAYS output the ## Overview section — it is mandatory
- Only output ## Key Discussion Points, ## Decisions Made, ## Action Items, ## Next Steps if the transcript has content for them
- Do NOT skip the ## heading line — even for a one-paragraph summary, the heading must appear

EXAMPLE of correct output format:
MEETING_TITLE: Introduction — John Smith
## Overview
John Smith introduced himself as a third-year computer science student at MIT. He mentioned his interest in machine learning and his hometown of Boston.

## Decisions Made
John decided to pursue an internship at a local startup over the summer rather than continuing his research position.

WRITING RULES:
- Introduce each speaker by full name on first mention, then use pronouns (he/she/they) naturally
- Do NOT start consecutive sentences with the same name
- Do NOT invent or infer anything not clearly stated in the transcript
- Reproduce all names, institutions, numbers, and statistics exactly as spoken
- Correct filler words (um, uh, repeated words) silently without changing meaning
- Return ONLY the formatted summary — no preamble, no sign-off

Transcript:
{truncated}

Summary:""",
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 1,
            },
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    if res.status_code != 200:
        detail = res.json().get("error") or res.text
        raise ValueError(f"HF API error: {detail}")

    try:
        data = res.json()
    except Exception:
        raise ValueError(f"HF returned non-JSON: {res.text[:300]}")

    print(f"HF raw response: {json.dumps(data)[:500]}")

    content = (
        (data.get("choices") or [{}])[0].get("message", {}).get("content")
        or (data.get("choices") or [{}])[0].get("text")
        or data.get("generated_text")
        or data.get("text")
        or data.get("content")
    )

    if not content:
        raise ValueError(f"Could not extract content from HF response: {json.dumps(data)[:300]}")

    return content.strip()


# ── Extract meeting title from raw model output ───────────
def extract_title(raw: str) -> tuple[str, str]:
    """
    Parses the MEETING_TITLE: line from the raw summary.
    Returns (title, summary_without_title_line).
    Falls back to 'Meeting Summary' if line is missing.
    """
    lines = raw.splitlines()
    title = "Meeting Summary"
    body_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("MEETING_TITLE:"):
            title = stripped[len("MEETING_TITLE:"):].strip().strip('"').strip("'")
        else:
            body_lines.append(line)
    return title, "\n".join(body_lines).strip()


# ── S3 upload ────────────────────────────────────────────
def _ascii_safe(text: str) -> str:
    """Replace common Unicode punctuation with ASCII equivalents, then strip remaining non-ASCII."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2012": "-",   # figure dash
        "\u2011": "-",   # non-breaking hyphen
        "\u2010": "-",   # hyphen
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for char, sub in replacements.items():
        text = text.replace(char, sub)
    return text.encode("ascii", errors="ignore").decode("ascii")

def upload_to_s3(summary: str, transcript: str, title: str = "Meeting Summary", uid: str = ""):
    try:
        # Scope to user: {uid}/summary-{ts}.txt  (or summary-{ts}.txt for legacy/anonymous)
        prefix = f"{uid}/" if uid else ""
        key = f"{prefix}summary-{int(time.time() * 1000)}.txt"
        s3_client.put_object(
            Bucket=os.getenv("AWS_BUCKET_NAME"),
            Key=key,
            Body=f"=== TITLE ===\n{title}\n\n=== SUMMARY ===\n{summary}\n\n=== TRANSCRIPT ===\n{transcript}",
            ContentType="text/plain",
            Metadata={"meeting-title": _ascii_safe(title)[:256]},
        )
        print(f"S3 uploaded: {key}")
        return key
    except Exception as e:
        print(f"S3 skipped: {e}")
        return None

# ── /start SSE endpoint ───────────────────────────────
@app.get("/start")
async def start(
    meet_link:    str = Query(..., alias="meetLink"),
    language:     str = Query("en-IN", alias="language"),
    meeting_name: str = Query("",     alias="meetingName"),
    uid:          str = Query("",     alias="uid"),
):
    # ── Validate uid for user isolation ───────────────────
    if not uid or not uid.strip():
        async def error_stream():
            yield sse("error", {"message": "User ID is required. Please sign in first."})
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_stream():
        yield sse("status", {"step": "joining", "message": "🤖 Creating Recall.ai bot…"})

        # Step 1: Create bot
        try:
            bot = await create_bot(meet_link)
            bot_id = bot["id"]
            yield sse("status", {"step": "joining", "message": f"✅ Bot created (ID: {bot_id}) — joining meeting now…"})
        except Exception as e:
            yield sse("error", {"message": f"Failed to create bot: {str(e)}"})
            return

        # Step 2: Poll until meeting ends
        recording_url = None
        last_status = None
        waiting_room_count = 0

        for i in range(120):
            await asyncio.sleep(5)

            try:
                data = await get_bot(bot_id)
            except Exception as e:
                yield sse("status", {"step": "joining", "message": f"⚠️ Poll failed: {e} — retrying…"})
                continue

            if i == 0:
                print(f"=== BOT DATA (first poll) ===\n{json.dumps(data, indent=2)}")

            changes = data.get("status_changes", [])
            latest = changes[-1] if changes else None
            recall_status = (latest or {}).get("code") or (latest or {}).get("status") or "unknown"

            if recall_status != last_status:
                last_status = recall_status
                print(f"[{i+1}] Status → {recall_status}")

            mapped = map_status(recall_status)
            yield sse("status", {"step": mapped["step"], "message": mapped["msg"]})

            if recall_status == "in_waiting_room":
                waiting_room_count += 1
                if waiting_room_count >= 9:
                    yield sse("error", {"message": "Bot was not admitted from the waiting room after 45 seconds."})
                    return
            else:
                waiting_room_count = 0

            if recall_status in TERMINAL_FAIL:
                yield sse("error", {"message": f"Bot failed: {recall_status}."})
                return

            if recall_status in TERMINAL_OK or (
                isinstance(data.get("status"), str) and data["status"] in ("call_ended", "done")
            ):
                yield sse("status", {"step": "processing", "message": "🎙 Meeting ended — waiting for recording upload…"})

                def safe_get_url(rec, shortcut_key):
                    try:
                        shortcuts = rec.get("media_shortcuts") or {}
                        shortcut = shortcuts.get(shortcut_key)
                        if shortcut and isinstance(shortcut, dict):
                            d = shortcut.get("data")
                            if d and isinstance(d, dict):
                                return d.get("download_url")
                    except Exception:
                        pass
                    return None

                for attempt in range(6):  # 6 × 15 s = 90 s max
                    await asyncio.sleep(15)
                    yield sse("status", {"step": "processing", "message": f"⏳ Waiting for recording… (attempt {attempt + 1}/6)"})

                    try:
                        fresh = await get_bot(bot_id)
                    except Exception as e:
                        print(f"[recording poll {attempt+1}] get_bot failed: {e}")
                        continue

                    recordings = fresh.get("recordings", [])
                    print(f"[recording poll {attempt+1}] recordings count={len(recordings)}")
                    if recordings:
                        print(f"[recording poll {attempt+1}] first recording keys: {list(recordings[0].keys())}")
                        print(f"[recording poll {attempt+1}] media_shortcuts: {json.dumps(recordings[0].get('media_shortcuts'), indent=2)}")

                    if recordings:
                        r = recordings[0]
                        recording_url = (
                            safe_get_url(r, "audio_mixed")
                            or safe_get_url(r, "video_mixed")
                            or r.get("download_url")
                            or r.get("url")
                        )
                        if recording_url:
                            print(f"[recording poll {attempt+1}] ✅ Got URL: {recording_url[:80]}…")
                            break
                        else:
                            print(f"[recording poll {attempt+1}] ⚠️ Recording found but no download_url yet")
                    else:
                        print(f"[recording poll {attempt+1}] ⚠️ No recordings in response yet")

                break  # exit main status-polling loop

        if not recording_url:
            yield sse("error", {"message": "No recording URL found after 90 seconds. Check your Recall.ai dashboard."})
            return

        # Step 3: Transcribe (Groq Whisper for English, Sarvam for Indian languages)
        lang_name = SUPPORTED_LANGUAGES.get(language, language)
        is_indian = language != "en-IN" and language in SUPPORTED_LANGUAGES
        if is_indian:
            yield sse("status", {"step": "transcribing", "message": f"🌐 Transcribing in {lang_name} via Sarvam AI…"})
        else:
            yield sse("status", {"step": "transcribing", "message": "🔊 Audio found. Converting & sending to Whisper…"})
        try:
            if is_indian:
                transcript = await sarvam_transcribe(recording_url, language)
            else:
                transcript = await transcribe(recording_url)
            yield sse("status", {"step": "transcribing", "message": f"✅ Transcript ready ({len(transcript)} chars)…"})
            yield sse("transcript", {"text": transcript})
        except Exception as e:
            yield sse("error", {"message": f"Transcription failed: {e}"})
            return

        # Step 4: Summarize
        yield sse("status", {"step": "summarizing", "message": "✨ Summarizing …"})
        try:
            raw_summary = await summarize(transcript)
        except Exception as e:
            print(f"Summarization error: {e}")
            yield sse("error", {"message": f"Summarization failed: {e}"})
            return

        # Extract meeting title and clean summary
        # User-supplied name takes priority over AI-detected title
        ai_title, summary = extract_title(raw_summary)
        meeting_title = meeting_name.strip() if meeting_name.strip() else ai_title
        print(f"Meeting title: {meeting_title} (source: {'user' if meeting_name.strip() else 'AI'})")  

        # Step 5: S3 — store under uid prefix for per-user isolation
        s3_key = upload_to_s3(summary, transcript, meeting_title, uid=uid)

        yield sse("done", {
            "summary":      summary,
            "transcript":   transcript,
            "s3Key":        s3_key,
            "language":     language,
            "langName":     lang_name,
            "meetingTitle": meeting_title,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── /languages ───────────────────────────────────────────
@app.get("/languages")
def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}

# ── /translate  (POST, JSON body) ───────────────────────────
class TranslateRequest(BaseModel):
    text:        str
    source_lang: str = "en-IN"
    target_lang: str

@app.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    if not SARVAM_API_KEY:
        return {"error": "SARVAM_API_KEY not configured."}
    if req.source_lang == req.target_lang:
        return {"translated_text": req.text}
    try:
        translated = await sarvam_translate(req.text, req.source_lang, req.target_lang)
        return {"translated_text": translated}
    except Exception as e:
        print(f"translate_endpoint error: {e}")
        return {"error": str(e)}

# ── /bot/:id debug endpoint ───────────────────────────────
@app.get("/bot/{bot_id}")
async def bot_status(bot_id: str):
    try:
        return await get_bot(bot_id)
    except Exception as e:
        return {"error": str(e)}

# ── /summaries  — list all saved summaries from S3 ────────
@app.get("/summaries")
def list_summaries(uid: str = Query("", alias="uid")):
    # Require uid for security — no listing without authentication
    if not uid or not uid.strip():
        return {"summaries": [], "error": "User ID is required. Please sign in."}
    
    try:
        bucket = os.getenv("AWS_BUCKET_NAME")
        # Scope the listing to this user's prefix
        prefix = f"{uid.strip()}/summary-"
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        items = []
        for obj in (resp.get("Contents") or []):
            key = obj["Key"]
            try:
                # Key may be "uid/summary-{ts_ms}.txt" or legacy "summary-{ts_ms}.txt"
                base = key.split("/")[-1]  # strip uid prefix
                ts_ms = int(base.replace("summary-", "").replace(".txt", ""))
                ts = ts_ms / 1000
            except Exception:
                ts = obj["LastModified"].timestamp()

            # Fetch S3 object metadata to get the stored meeting title
            meeting_title = None
            try:
                head = s3_client.head_object(Bucket=bucket, Key=key)
                metadata = head.get("Metadata", {})
                meeting_title = metadata.get("meeting-title") or None
            except Exception as e:
                print(f"head_object failed for {key}: {e}")

            items.append({
                "key": key,
                "timestamp": ts,
                "size": obj["Size"],
                "meetingTitle": meeting_title,
            })
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"summaries": items}
    except Exception as e:
        print(f"list_summaries error: {e}")
        return {"summaries": [], "error": str(e)}

# ── /summaries/{key}  — fetch a single summary from S3 ────
@app.get("/summaries/{key:path}")
def get_summary(key: str, uid: str = Query("", alias="uid")):
    # Require uid and validate key ownership — prevent cross-user access
    if not uid or not uid.strip():
        return {"error": "User ID is required. Please sign in."}
    
    # Ensure key belongs to this uid (security check)
    if not key.startswith(f"{uid.strip()}/"):
        return {"error": "Unauthorized: This summary does not belong to your account."}
    
    try:
        bucket = os.getenv("AWS_BUCKET_NAME")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")

        # Parse title (new format has === TITLE === block)
        meeting_title = "Meeting Summary"
        if "=== TITLE ===" in raw:
            title_part = raw.split("=== TITLE ===", 1)[1].split("\n\n===", 1)[0].strip()
            if title_part:
                meeting_title = title_part

        # Parse summary and transcript
        summary_text = raw
        transcript_text = ""
        if "=== SUMMARY ===" in raw and "=== TRANSCRIPT ===" in raw:
            parts = raw.split("\n\n=== TRANSCRIPT ===\n", 1)
            summary_text = parts[0].replace("=== TITLE ===\n" + meeting_title, "", 1)
            summary_text = summary_text.replace("=== SUMMARY ===\n", "", 1).strip()
            transcript_text = parts[1].strip() if len(parts) > 1 else ""

        try:
            # Key format: {uid}/summary-{ts_ms}.txt — extract just the filename
            base = key.split("/")[-1]  # "summary-1234567890.txt"
            ts_ms = int(base.replace("summary-", "").replace(".txt", ""))
            ts = ts_ms / 1000
        except Exception:
            ts = 0

        preview = next((l.strip() for l in summary_text.splitlines() if l.strip()), "")
        preview = preview.lstrip("#* ").replace("**", "")[:80]

        return {
            "key": key,
            "timestamp": ts,
            "preview": preview,
            "summary": summary_text,
            "transcript": transcript_text,
            "meetingTitle": meeting_title,
        }
    except Exception as e:
        print(f"get_summary error: {e}")
        return {"error": str(e)}

# ── /summaries/{key}/rename  — update meeting title ───────
class RenameRequest(BaseModel):
    title: str

@app.patch("/summaries/{key:path}/rename")
def rename_summary(key: str, req: RenameRequest, uid: str = Query("", alias="uid")):
    # Require uid and validate key ownership — prevent cross-user access
    if not uid or not uid.strip():
        return {"error": "User ID is required. Please sign in."}
    
    # Ensure key belongs to this uid (security check)
    if not key.startswith(f"{uid.strip()}/"):
        return {"error": "Unauthorized: This summary does not belong to your account."}
    
    try:
        bucket = os.getenv("AWS_BUCKET_NAME")
        new_title = req.title.strip() or "Meeting Summary"

        # Download existing file body
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8")

        # Replace or prepend the === TITLE === block
        if "=== TITLE ===" in raw and "\n\n=== SUMMARY ===" in raw:
            after_title = raw.split("\n\n=== SUMMARY ===", 1)[1]
            new_raw = f"=== TITLE ===\n{new_title}\n\n=== SUMMARY ==={after_title}"
        else:
            new_raw = f"=== TITLE ===\n{new_title}\n\n{raw}"

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=new_raw,
            ContentType="text/plain",
            Metadata={"meeting-title": _ascii_safe(new_title)[:256]},
        )
        print(f"Renamed {key} → {new_title}")
        return {"ok": True, "title": new_title}
    except Exception as e:
        print(f"rename_summary error: {e}")
        return {"error": str(e)}

# ── /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)