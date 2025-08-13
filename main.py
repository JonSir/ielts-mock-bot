import os, io, tempfile, time, random, re
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, Form, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from faster_whisper import WhisperModel
from scoring import metrics_from_text_and_times, bands_from_metrics

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000")
ELEVEN_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # can change later

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_BASE_URL, "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def permissions_policy(request: Request, call_next):
    resp = await call_next(request)
    # Allow mic and (hint) autoplay from this origin
    resp.headers["Permissions-Policy"] = "microphone=(self), autoplay=(self)"
    return resp

# Faster, smaller model for free hosting (OK for MVP)
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

PART1_TOPICS = {
    "home": [
        "Do you live in a house or an apartment?",
        "What do you like most about your home?",
        "Is there anything you would like to change about your home?"
    ],
    "work_study": [
        "Do you work or are you a student?",
        "Why did you choose your field?",
        "What do you find most interesting about it?"
    ],
    "hobbies": [
        "What do you like to do in your free time?",
        "How long have you had this hobby?",
        "Do you prefer to do it alone or with others?"
    ],
}
PART2_CARDS = [
    {
        "topic": "A market you have visited",
        "prompt": "Describe a market you have visited. You should say: where it is, when you went there, what you bought, and explain how you felt about the place.",
        "follow_up": "Do you think traditional markets will continue to be popular in the future?"
    },
    {
        "topic": "A journey you enjoyed",
        "prompt": "Describe a journey you enjoyed. You should say: where you went, why you went there, who you went with, and explain what you enjoyed about the journey.",
        "follow_up": "How has travel changed in your country in recent years?"
    }
]
PART3_THEMES = [
    {
        "theme": "Lifestyle and tradition",
        "questions": [
            "How do traditional markets influence community life?",
            "What are the advantages and disadvantages of modern supermarkets?",
            "How should cities balance tradition and modernization?"
        ]
    },
    {
        "theme": "Transport and mobility",
        "questions": [
            "How does improved transportation affect people’s work and leisure?",
            "What are some environmental impacts of increased travel?",
            "Should governments invest more in public transport? Why?"
        ]
    }
]

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
    SESSIONS[sid] = {
        "id": sid,
        "user_id": user_id,
        "part": 1,
        "p1_idx": 0,
        "p1_qs": p1_questions,
        "p2_card": card,
        "p2_spoke": False,
        "p3_idx": 0,
        "p3_theme": p3,
        "turns": [],
        "start_ts": time.time()
    }
    return SESSIONS[sid]

def summarize(scores: Dict[str,float], metrics: Dict[str,float]) -> str:
    lines = []
    f = scores["fluency"]; l = scores["lexical_resource"]; g = scores["grammar_range_accuracy"]; p = scores["pronunciation"]
    if f >= 6.5:
        lines.append("Fluency: You spoke at a steady pace with generally coherent development; fillers and pauses did not impede communication.")
    else:
        lines.append("Fluency: Pace and pausing sometimes disrupted flow; aim to extend answers and reduce fillers for smoother delivery.")
    if l >= 6.5:
        lines.append("Lexical resource: Good range with appropriate word choice; limited repetition and some topic-specific vocabulary.")
    else:
        lines.append("Lexical resource: Range was somewhat limited with repetition; try to vary expressions and use more precise terms.")
    if g >= 6.5:
        lines.append("Grammar: Mostly accurate with a mix of simple and some complex structures; few errors affecting meaning.")
    else:
        lines.append("Grammar: Frequent basic errors and limited variety; practice complex sentences and check subject–verb agreement.")
    if p >= 6.5:
        lines.append("Pronunciation: Generally clear with understandable rhythm; minor issues did not reduce intelligibility.")
    else:
        lines.append("Pronunciation: Clarity and rhythm varied; work on connected speech and reduce hesitations.")
    return " ".join(lines[:4])

@app.post("/api/tts")
async def tts(text: str = Query(..., max_length=1200), voice_id: str | None = None, model_id: str = "eleven_multilingual_v2"):
    if not ELEVEN_API_KEY:
        return JSONResponse({"error": "Missing ELEVENLABS_API_KEY"}, status_code=500)
    vid = voice_id or ELEVEN_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id, "voice_settings": {"stability": 0.6, "similarity_boost": 0.8}}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            return JSONResponse({"error": r.text}, status_code=r.status_code)
        return StreamingResponse(io.BytesIO(r.content), media_type="audio/mpeg")

@app.post("/api/upload")
async def upload(
    audio: UploadFile,
    session_id: str = Form(...),
    part: int = Form(...),
    question_id: str = Form(...),
    duration_ms: int = Form(0)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename or ".webm")[1]) as f:
        f.write(await audio.read())
        path = f.name
    seg_iter, info = whisper_model.transcribe(path, language="en")
    segs = list(seg_iter)
    text = " ".join(s.text.strip() for s in segs).strip()
    if segs:
        conf_vals = [1.0 - getattr(s, "no_speech_prob", 0.0) for s in segs]
        conf = max(0.0, min(1.0, sum(conf_vals) / len(conf_vals)))
    else:
        conf = 0.5
    os.remove(path)
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
        if too_short:
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
    summary = (
        ("Fluency: You spoke at a steady pace with generally coherent development; fillers and pauses did not impede communication."
         if bands["fluency"] >= 6.5 else
         "Fluency: Pace and pausing sometimes disrupted flow; aim to extend answers and reduce fillers for smoother delivery.")
        + " " +
        ("Lexical resource: Good range with appropriate word choice; limited repetition and some topic-specific vocabulary."
         if bands["lexical_resource"] >= 6.5 else
         "Lexical resource: Range was somewhat limited with repetition; try to vary expressions and use more precise terms.")
        + " " +
        ("Grammar: Mostly accurate with a mix of simple and some complex structures; few errors affecting meaning."
         if bands["grammar_range_accuracy"] >= 6.5 else
         "Grammar: Frequent basic errors and limited variety; practice complex sentences and check subject–verb agreement.")
        + " " +
        ("Pronunciation: Generally clear with understandable rhythm; minor issues did not reduce intelligibility."
         if bands["pronunciation"] >= 6.5 else
         "Pronunciation: Clarity and rhythm varied; work on connected speech and reduce hesitations.")
    )
    return {
        "result": {
            "overall": bands["overall"],
            "fluency": bands["fluency"],
            "lexical_resource": bands["lexical_resource"],
            "grammar_range_accuracy": bands["grammar_range_accuracy"],
            "pronunciation": bands["pronunciation"],
            "summary": summary,
            "recommendation": "Focus on extending answers with supporting details, vary vocabulary with precise terms, and practice complex sentences while maintaining clear rhythm."
        }
    }

@app.post("/telegram/webhook")
async def telegram_webhook(update: Dict[str, Any]):
    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        text = update["message"].get("text", "")
        if text.startswith("/start"):
            await send_webapp_button(chat_id, "Start Mock Speaking", f"{PUBLIC_BASE_URL}/webapp")
        else:
            await send_webapp_button(chat_id, "Start Mock Speaking", f"{PUBLIC_BASE_URL}/webapp")
    return {"ok": True}

async def send_webapp_button(chat_id: int, label: str, url: str):
    if not TELEGRAM_BOT_TOKEN: return
    kb = {
        "keyboard": [[{"text": label, "web_app": {"url": url}}]],
        "resize_keyboard": True, "is_persistent": True
    }
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": "Tap to begin your mock speaking test:", "reply_markup": kb}
        )

HTML = """
<!doctype html>
<html><head>
  <meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IELTS Mock Speaking</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 0; padding: 16px; background: #0b0f19; color: #fff; }
    .card { max-width: 760px; margin: 0 auto; background: #141a2a; border-radius: 12px; padding: 16px; }
    .q { font-size: 1.2rem; margin: 12px 0; }
    .status { color: #aab; margin: 8px 0; }
    #transcriptEdit { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334; background: #0e1320; color: #fff; }
    .row { display: flex; gap: 10px; align-items: center; }
    .btn { background: #2b6; border: none; color: #fff; padding: 10px 14px; border-radius: 8px; cursor: pointer; }
    .btn:disabled { background: #345; cursor: not-allowed; }
    .overlay { position: fixed; inset: 0; background: rgba(11,15,25,0.95); display: flex; align-items: center; justify-content: center; z-index: 1000; }
    .overlay .center { text-align: center; max-width: 640px; padding: 0 16px; }
    .small { font-size: 0.9rem; color: #aab; margin-top: 8px; }
  </style>
</head>
<body>
  <div id="overlay" class="overlay">
    <div class="center">
      <h2>Mock IELTS Speaking</h2>
      <p>Tap the button to enable sound and microphone.</p>
      <button id="startBtn" class="btn">Tap to Start</button>
      <div class="small">If it doesn’t work, close and open again, then tap this button.</div>
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

let sessionId = null;
let currentPart = 1;
let currentQuestionId = null;
let recStart = 0;
let mediaRecorder, chunks = [], stream;

function beep(ms = 200, freq = 880) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    g.gain.value = 0.05;
    o.connect(g); g.connect(ctx.destination);
    o.frequency.value = freq; o.start();
    setTimeout(() => { o.stop(); ctx.close(); }, ms);
  } catch (e) {}
}

async function speak(text, { onend } = {}) {
  try {
    const res = await fetch(`/api/tts?text=${encodeURIComponent(text)}`, { method: 'POST' });
    if (!res.ok) { console.error('TTS failed'); onend && onend(); return; }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => { URL.revokeObjectURL(url); onend && onend(); };
    const p = audio.play();
    if (p && p.catch) {
      p.catch(err => {
        console.warn('Autoplay blocked, continuing without audio:', err);
        onend && onend(); // avoid deadlock if audio can’t start
      });
    }
  } catch (e) {
    console.error('speak error', e);
    onend && onend();
  }
}

function selectMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/mpeg'
  ];
  if (!window.MediaRecorder) return '';
  for (const t of types) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

function extFromMime(m) {
  if (!m) return 'webm';
  if (m.includes('mp4')) return 'mp4';
  if (m.includes('mpeg')) return 'mp3';
  return 'webm';
}

async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert('Microphone permission is required. Please allow it and try again.');
    throw e;
  }
  const mimeType = selectMimeType();
  const options = mimeType ? { mimeType } : undefined;
  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    console.error('MediaRecorder init failed, trying without options', e);
    mediaRecorder = new MediaRecorder(stream);
  }
  chunks = [];
  mediaRecorder.ondataavailable = e => e.data && chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
    await uploadAnswer(blob, extFromMime(mediaRecorder.mimeType));
    stream.getTracks().forEach(t => t.stop());
  };
  recStart = Date.now();
  mediaRecorder.start();
}

function stopRecording() { try { mediaRecorder && mediaRecorder.stop(); } catch (e) {} }

function showTranscriptEditor(initialText, confidence) {
  const input = document.getElementById('transcriptEdit');
  input.value = initialText || '';
  const timer = document.getElementById('editTimer');
  let left = 5;
  timer.textContent = `${left}s to auto-confirm`;
  const iv = setInterval(() => {
    left -= 1;
    timer.textContent = `${left}s to auto-confirm`;
    if (left <= 0) {
      clearInterval(iv);
      confirmTranscript(input.value);
    }
  }, 1000);
  document.getElementById('confirmBtn').onclick = () => {
    clearInterval(iv);
    confirmTranscript(input.value);
  };
}

async function uploadAnswer(blob, ext = 'webm') {
  const fd = new FormData();
  fd.append('audio', blob, `answer.${ext}`);
  fd.append('session_id', sessionId);
  fd.append('part', currentPart);
  fd.append('question_id', currentQuestionId);
  fd.append('duration_ms', String(Date.now() - recStart));
  const res = await fetch('/api/upload', { method: 'POST', body: fd });
  const json = await res.json();
  showTranscriptEditor(json.transcript, json.confidence);
}

async function confirmTranscript(text) {
  const res = await fetch('/api/transcript/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, part: currentPart, question_id: currentQuestionId, transcript: text, duration_ms: (Date.now() - recStart) })
  });
  const next = await res.json();
  advanceExam(next);
}

function setQuestion(q) { document.getElementById('question').textContent = q || ''; }
function setStage(s) { document.getElementById('stage').textContent = s || ''; }

async function advanceExam(payload) {
  if (payload.result) {
    setStage('Test finished.');
    setQuestion(`Overall ${payload.result.overall} — Fluency ${payload.result.fluency}, Lexis ${payload.result.lexical_resource}, Grammar ${payload.result.grammar_range_accuracy}, Pron ${payload.result.pronunciation}. ${payload.result.summary}`);
    return;
  }
  if (payload.instruction && payload.part === 2) {
    setStage('Part 2: Preparation (60s)…');
    setQuestion(payload.cue_card);
    await speak(payload.instruction, { onend: async () => {
      beep(200);
      setTimeout(async () => {
        setStage('Part 2: Speak for up to 2 minutes…');
        beep(200);
        await speak('You may begin.', { onend: async () => {
          currentPart = 2; currentQuestionId = 'p2_main';
          await startRecording();
          setTimeout(() => { stopRecording(); beep(200); }, 120000);
        }});
      }, 60000);
    }});
    return;
  }
  if (payload.next_question) {
    const q = payload.next_question;
    currentPart = q.part; currentQuestionId = q.question_id;
    setStage(`Part ${q.part}`);
    setQuestion(q.text);
    await speak(q.text, { onend: async () => {
      beep(120);
      await startRecording();
      const limit = (q.part === 1) ? 40000 : 50000;
      setTimeout(() => { stopRecording(); }, limit);
    }});
  }
}

async function bootstrap() {
  const tgUserId = tg?.initDataUnsafe?.user?.id || 'anon';
  const res = await fetch('/api/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tg_user_id: tgUserId })
  });
  const data = await res.json();
  sessionId = data.session_id;
  currentPart = 1;
  currentQuestionId = data.next_question.question_id;
  setStage('Part 1');
  await speak(data.speak, { onend: async () => { await advanceExam({ next_question: data.next_question }); }});
}

// Require a tap to unlock audio + mic, then start
async function userStart() {
  try {
    // Unlock audio: play a nearly silent beep within user gesture
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(); const g = ctx.createGain();
    g.gain.value = 0.001; o.connect(g); g.connect(ctx.destination); o.start();
    setTimeout(() => { o.stop(); ctx.close(); }, 30);
  } catch (e) {}
  // Pre-ask mic permission so later startRecording is instant
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach(t => t.stop());
  } catch (e) {
    // User can still grant when we startRecording()
    console.warn('Pre-ask mic failed', e);
  }
  document.getElementById('overlay').style.display = 'none';
  await bootstrap();
}

document.getElementById('startBtn').addEventListener('click', userStart);
</script>
</body></html>
"""

@app.get("/webapp")
async def webapp():
    return HTMLResponse(HTML)

@app.get("/")
def health():
    return {"ok": True}
