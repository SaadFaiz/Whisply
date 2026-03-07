import asyncio
from sqlalchemy import case
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import deque
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
MODEL_NAME = "turbo"        # Whisper model variant
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BEAM_SIZE = 5
WORDS_PER_LINE = 15 * 2 #17 word per each line. 2 lines max per language
OPSET_VERSION = 16
USE_ONNX = False
FRAME_SIZE = 512 if SAMPLE_RATE == 16000 else 256

# VAD Configuration
SILENCE_DURATION = 2.0       # seconds of silence to trigger transcription
VAD_CHUNK_MS = 100           # process audio in 100ms chunks for VAD
VAD_THRESHOLD = 0.3          # speech probability threshold
MIN_SPEECH_DURATION = 0.1    # minimum seconds of speech to transcribe

# translation configuration
TRANSLATION_MODEL_NAME = "translategemma:4b"
TGT_LANG = "arb_Arab"
SRC_LANG = "eng_Latn" 

# ----------------------
# Load models
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading start")
startloading = time.perf_counter()

whisper_model = WhisperModel(
    MODEL_NAME,
    device=device,
    compute_type=COMPUTE_TYPE
)
whisperLoaded = True
print("WhisperModel loaded!")

SileroModel = load_silero_vad(onnx=USE_ONNX).to(device)
print("SileroModel loaded!")

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
print("Translation Model loaded!")
endloading = time.perf_counter()

print(f"Models loaded in {endloading - startloading:.2f} seconds")

# ----------------------
# Audio stream reader with VAD-based chunking
# ----------------------

async def audio_stream_reader(file_path: str, queue: asyncio.Queue, event_signal: asyncio.Event, queue_signal: asyncio.Queue):

    async def start_ffmpeg_process(file_path, start_time=0.0):
        t0 = time.perf_counter()

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

        t1 = time.perf_counter()
        print(f"[TIMER] FFmpeg spawn took: {t1 - t0:.4f}s")

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
                loop_t0 = time.perf_counter()

                startReading = time.perf_counter()

                signal_t0 = time.perf_counter()
                if event_signal.is_set():
                    signal_content = await queue_signal.get()
                    if signal_content["signal"] == "change_timestamp":
                        process = await start_ffmpeg_process(signal_content["start_time"])
                        signal_content["signal"] = None
                        event_signal.clear()
                        return
                signal_t1 = time.perf_counter()

                readData_t0 = time.perf_counter()
                data = await process.stdout.read(vad_chunk_size * 10)
                readData_t1 = time.perf_counter()

                if not data:
                    if len(accumulated_audio) > 0:
                        q_t0 = time.perf_counter()

                        print("[DEBUG] : queue got something line 117")
                        await queue.put((
                            chunk_index,
                            accumulated_audio,
                            global_time - len(accumulated_audio) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        ))

                        q_t1 = time.perf_counter()
                        print(f"[TIMER] Final queue.put took: {q_t1 - q_t0:.4f}s")

                        chunk_index += 1

                    print("[Reader] No data, retrying in 5s...")
                    await asyncio.sleep(5)
                    break

                buffer += data

                buffer_t0 = time.perf_counter()

                while len(buffer) >= vad_chunk_size:
                    vad_chunk = buffer[:vad_chunk_size]
                    buffer = buffer[vad_chunk_size:]

                    pre_buffer.extend(vad_chunk)

                    np_t0 = time.perf_counter()
                    audio_numpy = numpy.frombuffer(vad_chunk, dtype=numpy.float32).copy()
                    np_t1 = time.perf_counter()

                    if len(audio_numpy) != vad_frame_samples:
                        continue

                    torch_t0 = time.perf_counter()
                    audio_tensor = torch.from_numpy(audio_numpy).to(device)
                    torch_t1 = time.perf_counter()

                    vad_t0 = time.perf_counter()
                    speech_prob = SileroModel(audio_tensor, SAMPLE_RATE).item()
                    vad_t1 = time.perf_counter()

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

                                    q_t0 = time.perf_counter()

                                    print(f"[VAD] Sending chunk #{chunk_index}")
                                    await queue.put((chunk_index, accumulated_audio, start_time))

                                    q_t1 = time.perf_counter()
                                    print(f"[TIMER] queue.put took: {q_t1 - loop_t0:.4f}s")

                                    chunk_index += 1
                                else:
                                    print(f"[VAD] Speech too short ({speech_duration:.2f}s)")

                                accumulated_audio = b""
                                silence_duration = 0.0
                                is_speaking = False

                    global_time += chunk_duration

                buffer_t1 = time.perf_counter()

                endReading = time.perf_counter()

                loop_t1 = time.perf_counter()


        except asyncio.CancelledError:
            print("[Reader] Cancelled.")
            if process.returncode is None:
                process.kill()
            break

        finally:
            if process.returncode is None:
                process.kill()
            print("[DEBUG] : queue got something line 189")
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
            q1 = time.perf_counter()
            item = await queue.get()
            q2 = time.perf_counter()
            print("[Debug] took ", q2 - q1, " to get audio from queue")
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
            start_processing = time.perf_counter()

            try:
                start_whisper = time.perf_counter()
                segments, _ = whisper_model.transcribe(
                    audio_numpy,
                    language=VideoLanguage,
                    vad_filter=False,
                    beam_size=BEAM_SIZE,
                    word_timestamps=True
                )
                end_whisper = time.perf_counter()
                print(f"[Debug] Chunk {chunk_index} Whisper transcription done in {end_whisper - start_whisper:.6f}s")
            except Exception as e:
                print(f"[Error] Whisper transcription failed for chunk {chunk_index}: {e}")
                segments = []
                
            try:
                if not transcriptionStarted:
                    await websocket.send_text("start")
                    transcriptionStarted = True
                    print("[Debug] Sent 'start' to websocket")
            except Exception as e:
                print(f"[Error] Failed to send start signal: {e}")

            words = []

            cStart = time.perf_counter()
            sStart = time.perf_counter()
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
                                s1 = time.perf_counter()
                                print(f"[Debug] Frist line sent in {s1 - q1:.6f}s")
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
                            inputs = tokenizer(line['text'], return_tensors="pt").to(device)
                            translated_tokens = model.generate(
                                **inputs,
                                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG)
                            )
                            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
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
                            translated_text = ""
                except Exception as e:
                    print(f"[Error] Failed to process translation for chunk {chunk_index}: {e}")


            except Exception as e:
                print(f"[Error] Failed to process aligned segments for chunk {chunk_index}: {e}")
            sEnd = time.perf_counter()
            print(f"[Debug] Chunk {chunk_index} processing done in {sEnd - sStart:.6f}s")
            wStart = time.perf_counter()
            
            wEnd = time.perf_counter()
            print(f"[Debug] Chunk {chunk_index} line grouping done in {wEnd - wStart:.6f}s")
            cEnd = time.perf_counter()
            print(f"[Debug] Chunk {chunk_index} total processing done in {cEnd - cStart:.6f}s")
            end_processing = time.perf_counter()
            print(f"[Debug] Chunk {chunk_index} total processing time: {end_processing - start_processing:.6f}s")

        except Exception as e:
            print(f"[Transcription error in chunk {chunk_index}]: {e}")

#async def translation_worker(segment: str):
#     payload = {
#         "model": "translategemma:4b",
#         "prompt": f"Translate the following text to Arabic and output only the trasnlated Arabic text:\n\n{segment['text']}\n\n",
#         "stream": False
#     }
#     response =requests.post("http://localhost:11434/api/generate", json=payload)
#     TranslatedSegment = response.json().get("response")
#     return TranslatedSegment
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