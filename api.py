"""FastAPI backend. Run: uvicorn app.api:app --reload  (docs at /docs)"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from app import config, db
from app.agent import tools
from app.rag import indexer, pipeline
from app.security import create_token, decode_token

_state = {"retriever": None}


def get_retriever():
    if _state["retriever"] is None:
        if not config.CHUNKS_FILE.exists():
            indexer.build_index()
        from app.rag.retriever import HybridRetriever
        _state["retriever"] = HybridRetriever()
    return _state["retriever"]


def _reindex():
    indexer.build_index()
    if _state["retriever"]:
        _state["retriever"].reload()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="HR Policy Assistant API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- auth ----------
def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    payload = decode_token(authorization.split(" ", 1)[1], config.SECRET_KEY)
    user = payload and db.get_user(payload["sub"])
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


def require(*roles):
    def dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(403, "You do not have permission for this action")
        return user
    return dep


class LoginIn(BaseModel):
    username: str
    password: str


class ChatIn(BaseModel):
    question: str


class FeedbackIn(BaseModel):
    message_id: int
    rating: int  # 1 = helpful, -1 = not helpful
    comment: str = ""


class LeaveIn(BaseModel):
    leave_type: str
    start_date: str
    end_date: str
    reason: str = ""


class DecisionIn(BaseModel):
    approve: bool


class ReplyIn(BaseModel):
    reply: str


class SpeakIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginIn):
    user = db.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(401, "Incorrect username or password")
    return {"token": create_token({"sub": user["id"], "role": user["role"]}, config.SECRET_KEY,
                                  config.TOKEN_TTL_HOURS), "user": user}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return user


# ---------- chat ----------
@app.post("/chat")
def chat(body: ChatIn, user: dict = Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    history = []
    for m in db.get_history(user["id"], 3):
        history += [("user", m["question"]), ("assistant", m["answer"])]
    try:
        result = pipeline.ask(user, question, history, get_retriever())
    except RuntimeError as e:  # e.g. missing GROQ_API_KEY
        raise HTTPException(503, str(e))
    mid = db.add_message(user["id"], question, result["answer"], result["sources"], result["route"],
                         result["latency_ms"], result["ticket_id"])
    return {"message_id": mid, **result}


@app.get("/history")
def history(limit: int = 30, user: dict = Depends(current_user)):
    return db.get_history(user["id"], limit)


@app.post("/feedback")
def feedback(body: FeedbackIn, user: dict = Depends(current_user)):
    if body.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    db.add_feedback(body.message_id, user["id"], body.rating, body.comment)
    return {"ok": True}


# ---------- voice ----------
@app.post("/voice/transcribe")
async def transcribe(file: UploadFile = File(...), user: dict = Depends(current_user)):
    from app import voice
    try:
        return {"text": voice.transcribe(await file.read(), file.filename or "audio.wav")}
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/voice/speak")
def speak(body: SpeakIn, user: dict = Depends(current_user)):
    from app import voice
    try:
        return Response(content=voice.speak(body.text), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(503, f"Text-to-speech failed: {e}")


# ---------- leave ----------
@app.get("/leave/balance")
def leave_balance(user: dict = Depends(current_user)):
    return db.get_balances(user["id"])


@app.post("/leave/apply")
def leave_apply(body: LeaveIn, user: dict = Depends(current_user)):
    result = tools.apply_leave(user, body.leave_type, body.start_date, body.end_date, body.reason)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@app.get("/leave/requests")
def leave_requests(user: dict = Depends(current_user)):
    return db.list_leave_requests(user_id=user["id"])


@app.get("/leave/pending")
def leave_pending(user: dict = Depends(require("manager", "hr"))):
    return [r for r in db.list_leave_requests(status="pending") if r["user_id"] != user["id"]]


@app.post("/leave/{request_id}/decision")
def leave_decision(request_id: int, body: DecisionIn, user: dict = Depends(require("manager", "hr"))):
    result = tools.decide_leave(request_id, body.approve)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


# ---------- tickets ----------
@app.get("/tickets")
def tickets(all: bool = False, user: dict = Depends(current_user)):
    return db.list_tickets(None if (all and user["role"] == "hr") else user["id"])


@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: int, body: ReplyIn, user: dict = Depends(require("hr"))):
    db.resolve_ticket(ticket_id, body.reply)
    return {"ok": True}


# ---------- admin (HR only) ----------
@app.post("/admin/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...), min_role: str = Form("employee"),
                 user: dict = Depends(require("hr"))):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in {".pdf", ".md", ".txt"}:
        raise HTTPException(400, "Only .pdf, .md and .txt files are supported")
    if min_role not in config.ROLE_LEVEL:
        raise HTTPException(400, "min_role must be employee, manager or hr")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / name).write_bytes(await file.read())
    access = indexer.load_access()
    access[name] = min_role
    indexer.save_access(access)
    background.add_task(_reindex)
    return {"ok": True, "file": name, "min_role": min_role, "message": "Saved. Re-indexing in the background."}


@app.post("/admin/reindex")
def reindex(background: BackgroundTasks, user: dict = Depends(require("hr"))):
    background.add_task(_reindex)
    return {"ok": True, "message": "Re-indexing started."}


@app.get("/admin/analytics")
def analytics(user: dict = Depends(require("hr"))):
    return db.analytics()


@app.get("/admin/feedback")
def admin_feedback(user: dict = Depends(require("hr"))):
    return db.negative_feedback()
