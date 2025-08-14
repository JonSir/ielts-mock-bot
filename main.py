import os, io, tempfile, time, random, re
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, Form, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from faster_whisper import WhisperModel
from scoring import metrics_from_text_and_times, bands_from_metrics

# ====== Config ======
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "http://localhost:8000"
ELEVEN_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVEN_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# ====== App ======
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def permissions_policy(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Permissions-Policy"] = "microphone=(self), autoplay=(self)"
    resp.headers["Feature-Policy"] = "microphone 'self'; autoplay 'self'"
    return resp

# ====== Models / Data ======
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
PART1_TOPICS = {
    "home": ["Do you live in a house or an apartment?", "What do you like most about your home?", "Is there anything you would like to change about your home?"],
    "work_study": ["Do you work or are you a student?", "Why did you choose your field?", "What do you find most interesting about it?"],
    "hobbies": ["What do you like to do in your free time?", "How long have you had this hobby?", "Do you prefer to do it alone or with others?"],
}
PART2_CARDS = [{"topic": "A market you have visited", "prompt": "Describe a market you have visited. You should say: where it is, when you went there, what you bought, and explain how you felt about the place.", "follow_up": "Do you think traditional markets will continue to be popular in the future?"}, {"topic": "A journey you enjoyed", "prompt": "Describe a journey you enjoyed. You should say: where you went, why you went there, who you went with, and explain what you enjoyed about the journey.", "follow_up": "How has travel changed in your country in recent years?"}]
PART3_THEMES = [{"theme": "Lifestyle and tradition", "questions": ["How do traditional markets influence community life?", "What are the advantages and disadvantages of modern supermarkets?", "How should cities balance tradition and modernization?"]}, {"theme": "Transport and mobility", "questions": ["How does improved transportation affect people’s work and leisure?", "What are some environmental impacts of increased travel?", "Should governments invest more in public transport? Why?"]}]

def extract_keywords(text: str, k: int = 5):
    stop = set("i me my we our ours ourselves you your yours yourself yourselves he him his she her it its they them their what which who whom this that these those am is are was were be been being have has had do does did a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just should now".split())
    words = re.findall(r"[a-zA-Z']+", text.lower())
    freq: Dict[str,int] = {}
    for w in words:
        if len(w) < 4 or w in stop: continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]]

SESSIONS: Dict[str, Dict[str, Any]] = {}

def new_session(user_id: str) -> Dict[str, Any]:
    sid = f"s_{int(time.time()*1000)}_{random.randint(1000,9999)}"
    topics = random.sample(list(PART1_TOPICS.keys()), k=3)
    p1_questions = sum([[{"text": q, "topic": t} for q in PART1_TOPICS[t]] for t in topics], [])
    card = random.choice(PART2_CARDS)
    p3 = random.choice(PART3_THEMES)
    SESSIONS[sid] = {"id": sid, "user_id": user_id, "part": 1, "p1_idx": 0, "p1_qs": p1_questions, "p2_card": card, "p2_spoke": False, "p3_idx": 0, "p3_theme": p3, "turns": [], "start_ts": time.time()}
    return SESSIONS[sid]

# ====== API: TTS, Upload, Exam Flow ======
@app.post("/api/tts")
async def tts(text: str = Query(..., max_length=1200), voice_id: str | None = None, model_id: str | None = None):
    if not ELEVEN_API_KEY:
        return JSONResponse({"error": "Missing ELEVENLABS_API_KEY"}, status_code=500)

    vid = voice_id or ELEVEN_VOICE_ID
    mid = model_id or ELEVEN_MODEL_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": mid, "voice_settings": {"stability": 0.6, "similarity_boost": 0.8}}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code != 200:
                error_detail = r.text
                try:
                    error_json = r.json()
                    error_detail = error_json.get("detail", {}).get("message", r.text)
                except Exception: pass
                return JSONResponse({"error": f"ElevenLabs API Error: {r.status_code} - {error_detail}"}, status_code=r.status_code)
            return StreamingResponse(io.BytesIO(r.content), media_type="audio/mpeg")
    except httpx.RequestError as exc:
        return JSONResponse({"error": f"HTTP request failed: {exc}"}, status_code=500)

@app.post("/api/upload")
async def upload(audio: UploadFile, session_id: str = Form(...), part: int = Form(...), question_id: str = Form(...), duration_ms: int = Form(0)):
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await audio.read())
        path = f.name
    seg_iter, info = whisper_model.transcribe(path, language="en")
    segs = list(seg_iter)
    text = " ".join(s.text.strip() for s in segs).strip()
    conf = max(0.0, min(1.0, sum([1.0 - getattr(s, "no_speech_prob", 0.0) for s in segs]) / len(segs))) if segs else 0.5
    try: os.remove(path)
    except Exception: pass
    return {"transcript": text, "confidence": conf, "duration_ms": duration_ms}

@app.post("/api/session/start")
async def session_start(request: Request):
    data = await request.json() if request.headers.get("content-type","").startswith("application/json") else {}
    user_id = str(data.get("tg_user_id", "anon"))
    sess = new_session(user_id)
    intro = "Good morning, my name is Sarvarbek. In this mock IELTS Speaking test, I will ask you some questions. Let's begin with Part 1."
    first_q = sess["p1_qs"][0]["text"]
    return {"session_id": sess["id"], "speak": intro, "next_question": {"part": 1, "question_id": "p1_0", "text": first_q}}

@app.post("/api/transcript/confirm")
async def transcript_confirm(payload: Dict[str, Any]):
    sid = payload["session_id"]; part = int(payload["part"]); qid = payload["question_id"]; text = payload.get("transcript","").strip()
    duration_ms = int(payload.get("duration_ms", 0))
    sess = SESSIONS.get(sid)
    if not sess: return JSONResponse({"error": "session not found"}, status_code=404)
    sess["turns"].append({"part": part, "qid": qid, "transcript": text, "duration_ms": duration_ms})

    if part == 1:
        words = len(re.findall(r"[a-zA-Z']+", text))
        too_short = words < 35 or duration_ms < 12000
        idx = sess["p1_idx"]
        if too_short and not qid.endswith("_f"):
            return {"next_question": {"part": 1, "question_id": f"p1_{idx}_f", "text": "Could you tell me a bit more about that? For example, why is it important to you in your daily life?"}}
        sess["p1_idx"] += 1
        if sess["p1_idx"] < len(sess["p1_qs"]):
            nxt = sess["p1_qs"][sess["p1_idx"]]["text"]
            return {"next_question": {"part": 1, "question_id": f"p1_{sess['p1_idx']}", "text": nxt}}
        else:
            sess["part"] = 2
            prep = "Now, Part 2. I will give you a topic, and you will have one minute to prepare and then speak for one to two minutes."
            cue = sess["p2_card"]["prompt"]
            return {"instruction": prep, "cue_card": cue, "prep_seconds": 60, "speak_seconds": 120, "part": 2}

    elif part == 2:
        if not sess["p2_spoke"]:
            sess["p2_spoke"] = True
            return {"next_question": {"part": 2, "question_id": "p2_follow", "text": sess["p2_card"]["follow_up"]}}
        else:
            sess["part"] = 3
            p2_texts = [t["transcript"] for t in sess["turns"] if t["part"] == 2]
            kw = extract_keywords(" ".join(p2_texts), 5)
            opener = f"Now let's talk about some broader issues related to {', '.join(kw[:2])}."
            q = sess["p3_theme"]["questions"][0]
            return {"instruction": opener, "next_question": {"part": 3, "question_id": "p3_0", "text": q}}

    elif part == 3:
        sess["p3_idx"] += 1
        if sess["p3_idx"] < len(sess["p3_theme"]["questions"]):
            q = sess["p3_theme"]["questions"][sess["p3_idx"]]
            return {"next_question": {"part": 3, "question_id": f"p3_{sess['p3_idx']}", "text": q}}
        else:
            return await session_finish({"session_id": sid})

    return {"status": "ok"}

@app.post("/api/session/finish")
async def session_finish(payload: Dict[str, Any]):
    sid = payload["session_id"]
    sess = SESSIONS.get(sid)
    if not sess: return JSONResponse({"error": "session not found"}, status_code=404)
    all_text = " ".join(t["transcript"] for t in sess["turns"]).strip()
    total_speech_ms = sum(t.get("duration_ms", 0) for t in sess["turns"])
    m = metrics_from_text_and_times(all_text, speech_ms=total_speech_ms, pauses_ms=0)
    bands = bands_from_metrics(m, asr_conf=0.7, rate_stability=0.7)
    summary = (("Fluency: You spoke at a steady pace with generally coherent development; fillers and pauses did not impede communication." if bands["fluency"] >= 6.5 else "Fluency: Pace and pausing sometimes disrupted flow; aim to extend answers and reduce fillers for smoother delivery.") + " " + ("Lexical resource: Good range with appropriate word choice; limited repetition and some topic-specific vocabulary." if bands["lexical_resource"] >= 6.5 else "Lexical resource: Range was somewhat limited with repetition; try to vary expressions and use more precise terms.") + " " + ("Grammar: Mostly accurate with a mix of simple and some complex structures; few errors affecting meaning." if bands["grammar_range_accuracy"] >= 6.5 else "Grammar: Frequent basic errors and limited variety; practice complex sentences and check subject–verb agreement.") + " " + ("Pronunciation: Generally clear with understandable rhythm; minor issues did not reduce intelligibility." if bands["pronunciation"] >= 6.5 else "Pronunciation: Clarity and rhythm varied; work on connected speech and reduce hesitations."))
    return {"result": {"overall": bands["overall"], "fluency": bands["fluency"], "lexical_resource": bands["lexical_resource"], "grammar_range_accuracy": bands["grammar_range_accuracy"], "pronunciation": bands["pronunciation"], "summary": summary, "recommendation": "Focus on extending answers with supporting details, vary vocabulary with precise terms, and practice complex sentences while maintaining clear rhythm."}}

# ====== Telegram Webhook ======
@app.post("/telegram/webhook")
async def telegram_webhook(update: Dict[str, Any]):
    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        url = f"{PUBLIC_BASE_URL}/webapp?v=12" # Cache buster
        await send_webapp_button(chat_id, "Start Mock Speaking", url)
    return {"ok": True}

async def send_webapp_button(chat_id: int, label: str, url: str):
    if not TELEGRAM_BOT_TOKEN: return
    kb = {"keyboard": [[{"text": label, "web_app": {"url": url}}]], "resize_keyboard": True, "is_persistent": True}
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": "Tap to begin your mock speaking test:", "reply_markup": kb})

# ====== Web App (HTML + JS) ======
HTML = f"""
<!doctype html>
<html><head>
  <meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IELTS Mock Speaking</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; margin: 0; padding: 16px; background: #0b0f19; color: #fff; }}
    .card {{ max-width: 760px; margin: 0 auto; background: #141a2a; border-radius: 12px; padding: 16px; }}
    .q {{ font-size: 1.2rem; margin: 12px 0; }}
    .status {{ color: #aab; margin: 8px 0; }}
    #transcriptEdit {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334; background: #0e1320; color: #fff; }}
    .row {{ display: flex; gap: 10px; align-items: center; }}
    .btn {{ background: #2b6; border: none; color: #fff; padding: 10px 14px; border-radius: 8px; cursor: pointer; }}
    .btn:disabled {{ background: #345; cursor: not-allowed; }}
    .overlay {{ position: fixed; inset: 0; background: rgba(11,15,25,0.95); display: flex; align-items: center; justify-content: center; z-index: 1000; }}
    .overlay .center {{ text-align: center; max-width: 640px; padding: 0 16px; }}
    .small {{ font-size: 0.9rem; color: #aab; margin-top: 8px; }}
    .hint {{ font-size: 0.9rem; color: #9ac; margin-top: 10px; }}
    a.link {{ color: #7cf; }}
  </style>
</head>
<body>
  <div id="overlay" class="overlay">
    <div class="center">
      <h2>Mock IELTS Speaking</h2>
      <p id="overlayMsg">Tap the button to enable sound and microphone.</p>
      <button id="startBtn" class="btn">Tap to Start</button>
      <div class="small">If it doesn’t work, close and open again, then tap this button.</div>
      <div class="hint">If Telegram Desktop blocks the mic, open in your browser:</div>
      <div class="small"><a class="link" href="{PUBLIC_BASE_URL}/webapp?v=12&localTts=1" target="_blank" rel="noopener">Open in browser</a></div>
    </div>
  </div>

  <div class="card">
    <div id="stage" class="status">Initializing…</div>
    <div id="question" class="q"></div>
    <div class="row">
      <input id="transcriptEdit" placeholder="Transcript quick fix (5s window)" />
      <button id="confirmBtn" class="btn">Confirm</button>
      <div id="editTimer" class="status"></div>
    </div>
  </div>

<script>
const tg = window.Telegram?.WebApp;
tg && tg.expand();

// Read mode from URL: if you open /webapp?...&localTts=1 we use free browser voice
const params = new URLSearchParams(location.search);
const useLocalTTS = params.get('localTts') === '1';
const supportsSpeech = 'speechSynthesis' in window && typeof window.SpeechSynthesisUtterance !== 'undefined';

// Load available voices (needed on some browsers)
let voicesReady;
function loadVoices() {{
  if (voicesReady) return voicesReady;
  voicesReady = new Promise(resolve => {{
    let v = window.speechSynthesis?.getVoices?.() || [];
    if (v.length) return resolve(v);
    const timer = setTimeout(() => resolve(window.speechSynthesis.getVoices()), 1200);
    window.speechSynthesis.onvoiceschanged = () => {{
      clearTimeout(timer);
      resolve(window.speechSynthesis.getVoices());
    }};
  }});
  return voicesReady;
}}

function chooseEnglishVoice(voices) {{
  // Try to pick a clear English voice if possible
  let v = voices.find(x => /en-(US|GB)/i.test(x.lang) && /Google|Microsoft|Male/i.test(x.name))
       || voices.find(x => /en/i.test(x.lang))
       || voices[0];
  return v || null;
}}

async function speakLocal(text, options) {{
  const onend = options && options.onend;
  try {{
    if (!supportsSpeech) {{ onend && onend(); return; }}
    await loadVoices();
    const u = new SpeechSynthesisUtterance(text);
    const all = window.speechSynthesis.getVoices();
    const picked = chooseEnglishVoice(all);
    if (picked) u.voice = picked;
    u.lang = (picked && picked.lang) || 'en-US';
    u.rate = 1.0;
    u.pitch = 1.0;

    u.onend = () => {{ onend && onend(); }};
    u.onerror = () => {{ onend && onend(); }};
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }} catch (e) {{
    console.warn('Local TTS error', e);
    onend && onend();
  }}
}}

async function speak(text, options) {{
  const onend = options && options.onend;
  if (useLocalTTS && supportsSpeech) {{
    await speakLocal(text, options);
    return;
  }}

  try {{
    const res = await fetch(`/api/tts?text=${{encodeURIComponent(text)}}`, {{ method: 'POST' }});
    if (!res.ok) {{
      const errJson = await res.json().catch(() => ({{ error: "TTS client error" }}));
      console.warn('Server TTS error', res.status, errJson.error);
      if (supportsSpeech) {{ await speakLocal(text, options); }} else {{ onend && onend(); }}
      return;
    }}
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    let finished = false;
    const finish = () => {{
      if (finished) return;
      finished = true;
      try {{ URL.revokeObjectURL(url); }} catch(e) {{}}
      onend && onend();
    }};
    const fallback = setTimeout(finish, 1800);
    audio.onplaying = () => clearTimeout(fallback);
    audio.onended = finish;
    const p = audio.play();
    if (p && p.catch) {{ p.catch(err => {{ console.warn('Autoplay blocked', String(err)); finish(); }}); }}
  }} catch (e) {{
    console.warn('speak() fetch error', String(e));
    if (supportsSpeech) {{ await speakLocal(text, options); }} else {{ onend && onend(); }}
  }}
}}

let sessionId = null;
let currentPart = 1;
let currentQuestionId = null;
let recStart = 0;
let mediaRecorder, chunks = [], stream;
let hasRecordingStarted = false;

function beep(ms = 200, freq = 880) {{
  try {{
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    g.gain.value = 0.05;
    o.connect(g); g.connect(ctx.destination);
    o.frequency.value = freq; o.start();
    setTimeout(() => {{ o.stop(); ctx.close(); }}, ms);
  }} catch (e) {{}}
}}

function selectMimeType() {{
  const types = ['audio/webm;codecs=opus', 'audio/webm'];
  if (!window.MediaRecorder) return '';
  for (const t of types) {{ if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t; }}
  return '';
}}
function extFromMime(m) {{
  if (!m) return 'webm';
  if (m.includes('mp4')) return 'mp4';
  if (m.includes('mpeg')) return 'mp3';
  return 'webm';
}}

async function startRecording() {{
  if (hasRecordingStarted) return;
  hasRecordingStarted = true;
  console.log("Attempting to start recording...");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
    alert('This environment does not allow microphone access. Please open in Chrome.');
    console.error('getUserMedia not supported');
    throw new Error('getUserMedia unavailable');
  }}
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: true, noiseSuppression: true, autoGainControl: true }} }});
    console.log("Microphone stream acquired.");
  }} catch (e) {{
    alert('Microphone permission is required. Please allow it and try again.');
    console.error('getUserMedia error:', e);
    throw e;
  }}
  const mimeType = selectMimeType();
  const options = mimeType ? {{ mimeType }} : undefined;
  try {{
    mediaRecorder = options ? new MediaRecorder(stream, options) : new MediaRecorder(stream);
    console.log("MediaRecorder created with options:", options);
  }} catch (e) {{
    console.error("MediaRecorder creation failed, trying without options.", e);
    mediaRecorder = new MediaRecorder(stream);
  }}
  chunks = [];
  mediaRecorder.onstart = () => {{
    console.log("Event: mediaRecorder.onstart fired. State:", mediaRecorder.state);
  }};
  mediaRecorder.ondataavailable = e => {{
    if (e.data && e.data.size > 0) {{
      console.log(`Event: mediaRecorder.ondataavailable fired. Chunk size: ${{e.data.size}}`);
      chunks.push(e.data);
    }}
  }};
  mediaRecorder.onstop = async () => {{
    console.log(`Event: mediaRecorder.onstop fired. Total chunks: ${{chunks.length}}. State: ${{mediaRecorder.state}}`);
    try {{
      if (chunks.length === 0) {{
        console.warn("No data was recorded. Uploading empty blob.");
      }}
      const blob = new Blob(chunks, {{ type: mediaRecorder.mimeType || 'audio/webm' }});
      await uploadAnswer(blob, extFromMime(mediaRecorder.mimeType || ''));
    }} catch (err) {{
      console.error('Error in onstop handler:', String(err));
    }} finally {{
      try {{ stream.getTracks().forEach(t => t.stop()); }} catch (_){{}}
      hasRecordingStarted = false;
      console.log("Stream tracks stopped.");
    }}
  }};
  mediaRecorder.onerror = (e) => {{
    console.error("Event: mediaRecorder.onerror fired.", e);
  }};
  recStart = Date.now();
  try {{
    mediaRecorder.start(1000); // Request data every 1 second
    console.log("mediaRecorder.start(1000) called.");
  }} catch (e) {{
    console.error("mediaRecorder.start(1000) failed, trying without timeslice.", e);
    mediaRecorder.start();
  }}
}}

function stopRecording() {{
  console.log("Attempting to stop recording...");
  try {{
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
        mediaRecorder.stop();
        console.log("mediaRecorder.stop() called.");
    }} else {{
        console.warn("stopRecording called but recorder was not active. State:", mediaRecorder?.state);
    }}
  }}
  catch (e) {{ console.error('stopRecording error', String(e)); }}
}}

function showTranscriptEditor(initialText, confidence) {{
  console.log("Showing transcript editor with text:", initialText);
  const input = document.getElementById('transcriptEdit');
  input.value = initialText || '';
  const timer = document.getElementById('editTimer');
  let left = 5;
  timer.textContent = `${{left}}s`;
  const iv = setInterval(() => {{
    left -= 1;
    timer.textContent = `${{left}}s`;
    if (left <= 0) {{ clearInterval(iv); confirmTranscript(input.value); }}
  }}, 1000);
  document.getElementById('confirmBtn').onclick = () => {{
    clearInterval(iv);
    confirmTranscript(input.value);
  }};
}}

async function uploadAnswer(blob, ext = 'webm') {{
  console.log(`Uploading answer. Blob size: ${{blob.size}}, type: ${{blob.type}}`);
  const fd = new FormData();
  fd.append('audio', blob, `answer.${{ext}}`);
  fd.append('session_id', sessionId);
  fd.append('part', currentPart);
  fd.append('question_id', currentQuestionId);
  fd.append('duration_ms', String(Date.now() - recStart));
  try {{
    const res = await fetch('/api/upload', {{ method: 'POST', body: fd }});
    const text = await res.text();
    let json;
    try {{ json = JSON.parse(text); }}
    catch (e) {{
      console.error('Upload parse error', e, text?.slice(0, 200));
      alert('Upload succeeded but response could not be read. Please try again.');
      return;
    }}
    showTranscriptEditor(json.transcript, json.confidence);
  }} catch (err) {{
    console.error("Upload fetch failed", err);
    alert("Could not upload your answer. Please check your connection and try again.");
  }}
}}

async function confirmTranscript(text) {{
  console.log("Confirming transcript:", text.slice(0, 50));
  const res = await fetch('/api/transcript/confirm', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ session_id: sessionId, part: currentPart, question_id: currentQuestionId, transcript: text, duration_ms: (Date.now() - recStart) }})
  }});
  const next = await res.json();
  advanceExam(next);
}}

function setQuestion(q) {{ document.getElementById('question').textContent = q || ''; }}
function setStage(s) {{ document.getElementById('stage').textContent = s || ''; }}

async function askAndRecord(text, part, qid, maxMs) {{
  setStage(`Part ${{part}}`);
  setQuestion(text);
  currentPart = part; currentQuestionId = qid;
  let started = false;
  const startNow = async () => {{
    if (started) return;
    started = true;
    beep(120);
    await startRecording();
    setTimeout(() => stopRecording(), maxMs);
  }};
  const hardTimer = setTimeout(startNow, 1700);
  await speak(text, {{ onend: () => {{ clearTimeout(hardTimer); startNow(); }} }});
}}

async function advanceExam(payload) {{
  console.log("Advancing exam with payload:", payload);
  if (payload.result) {{
    setStage('Test finished.');
    setQuestion(`Overall ${{payload.result.overall}} — Fluency ${{payload.result.fluency}}, Lexis ${{payload.result.lexical_resource}}, Grammar ${{payload.result.grammar_range_accuracy}}, Pron ${{payload.result.pronunciation}}. ${{payload.result.summary}}`);
    return;
  }}
  if (payload.instruction && payload.part === 2) {{
    setStage('Part 2: Preparation (60s)…');
    setQuestion(payload.cue_card);
    await speak(payload.instruction, {{ onend: async () => {{
      beep(200);
      setTimeout(async () => {{
        setStage('Part 2: Speak for up to 2 minutes…');
        beep(200);
        currentPart = 2; currentQuestionId = 'p2_main';
        await askAndRecord('You may begin.', 2, 'p2_main', 120000);
      }}, 60000);
    }}}});
    return;
  }}
  if (payload.next_question) {{
    const q = payload.next_question;
    const limit = (q.part === 1) ? 40000 : 50000;
    await askAndRecord(q.text, q.part, q.question_id, limit);
  }}
}}

async function bootstrap() {{
  const tgUserId = tg?.initDataUnsafe?.user?.id || 'anon';
  const res = await fetch('/api/session/start', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ tg_user_id: tgUserId }})
  }});
  const data = await res.json();
  sessionId = data.session_id;
  currentPart = 1;
  currentQuestionId = data.next_question.question_id;
  await speak(data.speak, {{ onend: async () => {{ await advanceExam({{ next_question: data.next_question }}); }}}});
}}

async function userStart() {{
  const msg = document.getElementById('overlayMsg');
  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  msg.textContent = 'Requesting microphone permission…';

  try {{
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(); const g = ctx.createGain();
    g.gain.value = 0.001; o.connect(g); g.connect(ctx.destination); o.start();
    setTimeout(() => {{ o.stop(); ctx.close(); }}, 30);
  }} catch (e) {{}}

  let granted = false;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
    msg.textContent = 'This app cannot access your microphone here. Try "Open in browser".';
    btn.disabled = false;
    return;
  }}
  try {{
    const s = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    s.getTracks().forEach(t => t.stop());
    granted = true;
  }} catch (e) {{}}

  if (!granted) {{
    msg.textContent = 'Microphone permission is required. Tap again and press "Allow".';
    btn.disabled = false;
    return;
  }}

  document.getElementById('overlay').style.display = 'none';
  await bootstrap();
}}

document.getElementById('startBtn').addEventListener('click', userStart);
</script>
</body></html>
"""

@app.get("/webapp")
async def webapp():
    return HTMLResponse(HTML)

@app.get("/")
def health():
    return {"ok": True, "base_url": PUBLIC_BASE_URL}
