import os
import asyncio
import io
import subprocess
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import torch
import wave

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Configuration
# ----------------------
CHUNK_DURATION = 8          # seconds per chunk
OVERLAP = 1                 # seconds overlap between chunks
SAMPLE_RATE = 16000         # audio sample rate
BYTES_PER_SAMPLE = 2        # 16-bit PCM
CHANNELS = 1
QUEUE_MAXSIZE = 5           # queue depth for backpressure
MODEL_NAME = "medium.en"       # large, large-v2, medium, small, etc.
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BEAM_SIZE = 10               # higher beam for accuracy
WORDS_PER_LINE = 10          # words per subtitle line

# ----------------------
# Load Whisper model
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(
    MODEL_NAME,
    device=device,
    compute_type=COMPUTE_TYPE
    )
print("Model loaded!")
# ----------------------
# Audio stream reader
# ----------------------
async def audio_stream_reader(stream_url: str, queue: asyncio.Queue):
    cmd = [
        "ffmpeg",
        "-re",
        "-tls_verify", "0",
        "-i", stream_url,
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-f", "wav", "pipe:1"
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL  # reduce noise
    )

    try:
        buffer = b""
        header_skipped = False
        chunk_size = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_DURATION
        overlap_size = SAMPLE_RATE * BYTES_PER_SAMPLE * OVERLAP
        global_time = 0.0

        while True:
            data = await process.stdout.read(4096)
            if not data:
                break
            buffer += data
            if not header_skipped and len(buffer) > 44:
                buffer = buffer[44:]
                header_skipped = True
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                await queue.put((chunk, global_time))
                buffer = buffer[chunk_size - overlap_size:]
                global_time += (CHUNK_DURATION - OVERLAP)
    except asyncio.CancelledError:
        print("[Reader] Cancelled.")
    finally:
        if process.returncode is None:
            process.kill()
        await queue.put(None)


# ----------------------
# Transcriber worker
# ----------------------
async def transcriber_worker(queue: asyncio.Queue, websocket: WebSocket, VideoLanguage: str):
    sent_buffer = []  # keep last segments for deduplication

    while True:
        item = await queue.get()
        if item is None:
            break
        chunk, start_time = item
        audio_buf = io.BytesIO()
        with wave.open(audio_buf, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(chunk)
        audio_buf.seek(0)

        try:
            segments, _ = model.transcribe(
                audio_buf,
                language=VideoLanguage,
                vad_filter=False,
                beam_size=BEAM_SIZE,
                word_timestamps=True
            )
            words = []
            for seg in segments:
                # filter by confidence
                if seg.no_speech_prob > 0.8 or seg.avg_logprob < -3.0 or seg.compression_ratio > 3.0:
                    continue
                for w in getattr(seg, 'words', []):
                    word_text = w.word.strip()
                    if not word_text:
                        continue
                    word_start = (w.start or seg.start) + start_time
                    word_end = (w.end or (w.start or seg.start) + 0.5) + start_time
                    key = (word_text.lower(), round(word_start, 2))
                    # dedupe
                    if key in sent_buffer:
                        continue
                    sent_buffer.append(key)
                    words.append({'text': word_text, 'start': word_start, 'end': word_end})
            # send grouped lines
            line, line_start = [], None
            for w in words:
                if not line:
                    line_start = w['start']
                line.append(w)
                if len(line) >= WORDS_PER_LINE or w['text'].endswith(('.', '!', '?')):
                    line_end = line[-1]['end']
                    txt = ' '.join(x['text'] for x in line)
                    await websocket.send_json({'start': line_start, 'end': line_end, 'text': txt})
                    line = []
            if line:
                line_end = line[-1]['end']
                txt = ' '.join(x['text'] for x in line)
                await websocket.send_json({'start': line_start, 'end': line_end, 'text': txt})
        except Exception as e:
            print(f"[Transcription error] {e}")

# ----------------------
# WebSocket endpoint
# ----------------------
print(f"Loading Whisper model '{MODEL_NAME}' on {device}...")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    reader = None
    worker = None

    try:
        stream_url = await websocket.receive_text()
        VideoLanguage = await websocket.receive_text()
        reader = asyncio.create_task(audio_stream_reader(stream_url, queue))
        worker = asyncio.create_task(transcriber_worker(queue, websocket, VideoLanguage))
        await asyncio.gather(reader, worker)
    except Exception as e:
        print(f"[WebSocket error] {e}")
    finally:
        # Clean up
        if reader:
            reader.cancel()
        if worker:
            worker.cancel()
        await websocket.close()
        while not queue.empty():
            queue.get_nowait()
        print("[Cleanup] WebSocket closed and resources cleared.")


# ----------------------
# HTTP endpoints
# ----------------------
@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)