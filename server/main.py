import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from collections import deque
from google import genai
from dotenv import load_dotenv
import os
import torch
import time
import numpy


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
SAMPLE_RATE = 16000          # audio sample rate
BYTES_PER_SAMPLE = 4         
CHANNELS = 1
QUEUE_MAXSIZE = 5
WORDS_PER_LINE = 15 * 2 #17 word per each line. 2 lines max per language
OPSET_VERSION = 16
USE_ONNX = False
FRAME_SIZE = 512 if SAMPLE_RATE == 16000 else 256

# ASR Configuration
MODEL_NAME = "turbo"        # Whisper model variant
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BEAM_SIZE = 5


# VAD Configuration
SILENCE_DURATION = 2.0       # seconds of silence to trigger transcription
VAD_CHUNK_MS = 100           # process audio in 100ms chunks for VAD
VAD_THRESHOLD = 0.3          # speech probability threshold
MIN_SPEECH_DURATION = 0.1    # minimum seconds of speech to transcribe

# translation configuration
TRANSLATION_MODEL_NAME = "gemma-3-27b-it"
TGT_LANG = "arb_Arab"
SRC_LANG = "eng_Latn" 

# ENV variables
load_dotenv()
GOOGLEAPIKEY = os.getenv("GOOGLE_API_KEY")
# ----------------------
# Load models
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading start")

whisper_model = WhisperModel(
    MODEL_NAME,
    device=device,
    compute_type=COMPUTE_TYPE
)
print("WhisperModel loaded!")

SileroModel = load_silero_vad(onnx=USE_ONNX).to(device)
print("SileroModel loaded!")

GenaiClient = genai.Client(api_key=GOOGLEAPIKEY)
print("Client loaded!")

# ----------------------
# Audio stream reader with VAD-based chunking
# ----------------------

async def audio_stream_reader(file_path: str, queue: asyncio.Queue, event_signal: asyncio.Event, queue_signal: asyncio.Queue):

    async def start_ffmpeg_process(file_path, start_time=0.0):
        print("Starting FFmpeg process (native asyncio)...")

        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", file_path,
            "-vn",
            "-sn",
            "-af", "volume=2.0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "pipe:1"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        return process

    while True:
        try:
            print("starting ffmpeg process...")
            process = await start_ffmpeg_process(file_path)
            print("ffmpeg process started.")

            buffer = b""
            accumulated_audio = b""
            silence_duration = 0.0
            is_speaking = False
            global_time = 0.0
            chunk_index = 0
            start_time = 0.0

            vad_frame_samples = FRAME_SIZE
            vad_chunk_size = vad_frame_samples * BYTES_PER_SAMPLE

            pre_buffer = deque(maxlen=int(SAMPLE_RATE * BYTES_PER_SAMPLE * 2)) 

            print("[Reader] Reading audio stream...")

            while True:
                if event_signal.is_set():
                    signal_content = await queue_signal.get()
                    if signal_content["signal"] == "change_timestamp":
                        process = await start_ffmpeg_process(signal_content["start_time"])
                        signal_content["signal"] = None
                        event_signal.clear()
                        return

                data = await process.stdout.read(vad_chunk_size * 10)

                if not data:
                    if len(accumulated_audio) > 0:
                        await queue.put((
                            chunk_index,
                            accumulated_audio,
                            global_time - len(accumulated_audio) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        ))
                        chunk_index += 1

                    print("[Reader] No data, retrying in 5s...")
                    await asyncio.sleep(5)
                    break

                buffer += data

                while len(buffer) >= vad_chunk_size:
                    vad_chunk = buffer[:vad_chunk_size]
                    buffer = buffer[vad_chunk_size:]

                    pre_buffer.extend(vad_chunk)

                    audio_numpy = numpy.frombuffer(vad_chunk, dtype=numpy.float32).copy()

                    if len(audio_numpy) != vad_frame_samples:
                        continue

                    audio_tensor = torch.from_numpy(audio_numpy).to(device)

                    speech_prob = SileroModel(audio_tensor, SAMPLE_RATE).item()

                    chunk_duration = len(vad_chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                    if speech_prob >= VAD_THRESHOLD:
                        if not is_speaking:
                            print(f"[VAD] Speech started at {global_time:.2f}s (prob={speech_prob:.3f})")
                            is_speaking = True
                            accumulated_audio = bytes(pre_buffer)

                        accumulated_audio += vad_chunk
                        silence_duration = 0.0

                    else:
                        if is_speaking:
                            silence_duration += chunk_duration
                            accumulated_audio += vad_chunk

                            if silence_duration >= SILENCE_DURATION:
                                speech_duration = len(accumulated_audio) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                                if speech_duration >= MIN_SPEECH_DURATION:
                                    start_time = global_time - speech_duration
                                    print(f"[VAD] Sending chunk #{chunk_index}")
                                    await queue.put((chunk_index, accumulated_audio, start_time))
                                    chunk_index += 1
                                else:
                                    print(f"[VAD] Speech too short ({speech_duration:.2f}s)")

                                accumulated_audio = b""
                                silence_duration = 0.0
                                is_speaking = False

                    global_time += chunk_duration

        except asyncio.CancelledError:
            print("[Reader] Cancelled.")
            if process.returncode is None:
                process.kill()
            break

        finally:
            if process.returncode is None:
                process.kill()
            await queue.put(None)

# ----------------------
# Transcriber worker
# ----------------------
async def transcriber_worker(queue: asyncio.Queue, websocket: WebSocket, VideoLanguage: str):
    transcriptionStarted = False
    lines = []

    counter = 0

    async def generate_line_id():
        nonlocal counter
        counter += 1
        line_id = (int(time.time() * 1000) + counter).to_bytes(8, byteorder='big').hex()
        return line_id 
        
    async def add_id_to_line(line):
        if 'id' not in line or line['id'] is None:
            line['id'] = await generate_line_id()
        return line
        
    async def send_ws(route, content):
        try:
            match route: 
                case "translation":
                    translation = {
                        'id': content['id'],
                        'translated_text': content['translated_text']
                    }
                    await websocket.send_json({"route": "translation", "content": translation})
                case "transcription":
                    line = await add_id_to_line(content)
                    lines.append(line)
                    await websocket.send_json({"route": "transcription", "content": line})
                    print(f"[WebSocket] Sent message: {line}")
        except Exception as e:
            print(f"[Error] Failed to send WebSocket message: {e}")
            
    while True:
        try:
            item = await queue.get()
        except Exception as e:
            print(f"[Error] Failed to get item from queue: {e}")
            continue

        if item is None:
            print("[Debug] Queue returned None, exiting worker")
            break

        try:
            chunk_index, chunk_bytes, start_time = item
            chunk_duration = len(chunk_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
            print(f"[Debug] Processing chunk #{chunk_index} ({chunk_duration:.2f}s)")
        except Exception as e:
            print(f"[Error] Failed to unpack chunk: {e}")
            continue

        try:
            audio_numpy = numpy.frombuffer(chunk_bytes, dtype=numpy.float32).copy()
            print("[Debug] WAV buffer created")
        except Exception as e:
            print(f"[Error] Failed to create WAV buffer: {e}")

        try:
            # ── Whisper transcription ──
            try:
                segments, _ = whisper_model.transcribe(
                    audio_numpy,
                    language=VideoLanguage,
                    vad_filter=False,
                    beam_size=BEAM_SIZE,
                    word_timestamps=True
                )
            except Exception as e:
                print(f"[Error] Whisper transcription failed for chunk {chunk_index}: {e}")
                segments = []
                
            # ── WS "start" signal ──
            try:
                if not transcriptionStarted:
                    await websocket.send_text("start")
                    transcriptionStarted = True
                    print("[Debug] Sent 'start' to websocket")
            except Exception as e:
                print(f"[Error] Failed to send start signal: {e}")

            # ── Segment / line building  ──
            try:
                line = {
                    'chunk': chunk_index,
                    "text" : "",
                    "start" : start_time,
                    "end" : None,
                    "words" : []
                }
                for seg in segments:
                    line["words"].extend(seg.words)
                    line["text"] =  line["text"] + " " + seg.text
                    if(len(seg.text) < WORDS_PER_LINE or any(p in seg.text for p in ['.', '!', '?']) == False):
                        continue
                    segWords = []
                    lineWordStartIndex = 0
                    print(f"[Debug][{seg.start} : {seg.end}] : " , seg.text)
                    for i, w in enumerate(line["words"]):
                        try:
                            word_text = w.word.strip()
                            segWords.append(word_text)
                            if not word_text:
                                continue
                            if(word_text.endswith(('.', '!', '?'))):
                                lineSent = {
                                    'chunk': chunk_index,
                                    'start': line["words"][lineWordStartIndex].start + start_time, 
                                    'end': w.end + start_time, 
                                    'text': ' '.join(segWords)
                                }
                                await send_ws("transcription",lineSent)
                                await asyncio.sleep(0)
                                print("[Debug] Line sent")
                                segWords = []
                                lineWordStartIndex = i + 1
                                continue
                            if len(segWords) >= WORDS_PER_LINE:
                                if(i >= len(line["words"])-6):
                                    continue
                                lineSent = {
                                    'chunk': chunk_index,
                                    'start': line["words"][lineWordStartIndex].start + start_time, 
                                    'end': w.end + start_time, 
                                    'text': ' '.join(segWords)
                                }
                                await send_ws("transcription",lineSent)
                                await asyncio.sleep(0)
                                print("[Debug] Line sent")
                                segWords = []
                                lineWordStartIndex = i + 1

                        except Exception as e:
                            print(f"[Error] Failed processing word in chunk {chunk_index}: {e}")

                    if segWords:
                        try:
                            flush_start_word = line["words"][lineWordStartIndex] if lineWordStartIndex < len(line["words"]) else line["words"][-1]
                            flush_end_word   = line["words"][-1]
                            lineSent = {
                                'chunk': chunk_index,
                                'start': flush_start_word.start + start_time,
                                'end':   flush_end_word.end + start_time,
                                'text':  ' '.join(w for w in segWords if w)
                            }
                            await send_ws("transcription",lineSent)
                            await asyncio.sleep(0)
                            print("[Debug] Flushed remaining segWords after word loop")
                        except Exception as e:
                            print(f"[Error] Failed flushing remaining segWords for chunk {chunk_index}: {e}")

                    line = {
                        'chunk': chunk_index,
                        "text" : "",
                        "start" : start_time,
                        "end" : None,
                        "words" : []
                    }

                if line["words"]:
                    try:
                        segWords = []
                        lineWordStartIndex = 0
                        for i, w in enumerate(line["words"]):
                            try:
                                word_text = w.word.strip()
                                segWords.append(word_text)
                                if not word_text:
                                    continue
                                if word_text.endswith(('.', '!', '?')):
                                    lineSent = {
                                        'chunk': chunk_index,
                                        'start': line["words"][lineWordStartIndex].start + start_time,
                                        'end':   w.end + start_time,
                                        'text':  ' '.join(segWords)
                                    }
                                    await send_ws("transcription",lineSent)
                                    await asyncio.sleep(0)
                                    print("[Debug] Flushed remaining line (post-loop, punctuation)")
                                    segWords = []
                                    lineWordStartIndex = i + 1
                                    continue
                                if len(segWords) >= WORDS_PER_LINE:
                                    lineSent = {
                                        'chunk': chunk_index,
                                        'start': line["words"][lineWordStartIndex].start + start_time,
                                        'end':   w.end + start_time,
                                        'text':  ' '.join(segWords)
                                    }
                                    await send_ws("transcription",lineSent)
                                    await asyncio.sleep(0)
                                    print("[Debug] Flushed remaining line (post-loop, length)")
                                    segWords = []
                                    lineWordStartIndex = i + 1
                            except Exception as e:
                                print(f"[Error] Failed flushing remaining line word in chunk {chunk_index}: {e}")
                        if segWords:
                            try:
                                flush_start_word = line["words"][lineWordStartIndex] if lineWordStartIndex < len(line["words"]) else line["words"][-1]
                                flush_end_word   = line["words"][-1]
                                lineSent = {
                                    'chunk': chunk_index,
                                    'start': flush_start_word.start + start_time,
                                    'end':   flush_end_word.end + start_time,
                                    'text':  ' '.join(w for w in segWords if w)
                                }
                                await send_ws("transcription",lineSent)
                                await asyncio.sleep(0)
                                print("[Debug] Flushed tail of remaining line (post-loop)")
                            except Exception as e:
                                print(f"[Error] Failed flushing tail of remaining line for chunk {chunk_index}: {e}")
                    except Exception as e:
                        print(f"[Error] Failed flushing remaining line for chunk {chunk_index}: {e}")

                print(f"lines for chunk {chunk_index}: {lines}")

                # ────────────────────── TRANSLATION LAYER ───────────────────────
                try:
                    for line in lines:
                        try:
                            prompt = f"""You are an expert {SRC_LANG} to {TGT_LANG} subtitle translator. Translate the "Current Line" into natural {TGT_LANG}. Rules:1. Output strictly the {TGT_LANG} translation of the Current Line and nothing else.2. Use the "Previous Line" strictly to understand context, pronouns, and gender, but DO NOT translate it.3. If the Current Line is incomplete, translate it as an incomplete thought.Previous Line: "{lines[lines.index(line)-1]['text'] if lines.index(line) > 0 else ''}"Current Line: "{line['text']}"next Line: "{lines[lines.index(line)+1]['text'] if lines.index(line) < len(lines)-1 else ''}"{TGT_LANG} translation:"""
        
                            response = GenaiClient.models.generate_content(
                                model=TRANSLATION_MODEL_NAME,
                                contents=prompt
                            )
                            translated_text = response.text.strip()
                            if translated_text:
                                translation = {
                                    'id': line['id'],
                                    'translated_text': translated_text
                                }
                                await send_ws("translation", translation)
                                await asyncio.sleep(0)
                                print(f"[Debug] Translation sent for line in chunk {chunk_index}")
                        except Exception as e:
                            print(f"[Error] Translation failed for line in chunk {chunk_index}: {e}")

                except Exception as e:
                    print(f"[Error] Failed to process translation for chunk {chunk_index}: {e}")

            except Exception as e:
                print(f"[Error] Failed to process aligned segments for chunk {chunk_index}: {e}")

        except Exception as e:
            print(f"[Transcription error in chunk {chunk_index}]: {e}")

# ----------------------
# WebSocket endpoint
# ----------------------
print(f"Loading Whisper model '{MODEL_NAME}' on {device}...")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    queue_signal = asyncio.Queue(maxsize=1)
    event_signal = asyncio.Event()
    reader = None
    worker = None
    try:
        file_path = await websocket.receive_text()
        VideoLanguage = await websocket.receive_text()
        reader = asyncio.create_task(audio_stream_reader(file_path, queue, event_signal, queue_signal))
        worker = asyncio.create_task(transcriber_worker(queue, websocket, VideoLanguage))
        await asyncio.gather(reader, worker)
    except Exception as e:
        print(f"[WebSocket error] {e}")
    finally:
        if reader:
            reader.cancel()
        if worker:
            worker.cancel()
        await websocket.close()
        while not queue.empty():
            queue.get_nowait()
        print("[Cleanup] WebSocket closed and resources cleared.")

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)