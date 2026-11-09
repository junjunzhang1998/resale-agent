import asyncio
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import research

HERE = Path(__file__).parent

# Graders wait up to 5 minutes. The research phase and the Writer have their own timeouts in
# research.py; this is the backstop for the whole answer.
ANSWER_TIMEOUT = 240

# --- Session Store ---

# session_id -> {"turns": [{role, content}], "clarifications": int}. In-memory, single process:
# deploy with one Cloud Run instance so a clarifying answer reaches the same session.
sessions: dict[str, dict] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    # Same shape as Project 1, plus memo: the structured pricing result for the page (None when
    # the agent asks a clarifying question or can't price the item).
    response: str
    session_id: str
    tool_calls: list[dict]
    memo: dict | None = None


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    # no-cache: browsers otherwise keep serving an old page after a redeploy
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.setdefault(session_id, {"turns": [], "clarifications": 0})
    session["turns"].append({"role": "user", "content": request.message})

    try:
        result = await asyncio.wait_for(research.answer(session["turns"], session["clarifications"]), ANSWER_TIMEOUT)
    except Exception as e:
        # Auth, quota, a model that won't stop: show it in the chat, not as a 500.
        result = {"response": f"Research failed: {type(e).__name__}: {str(e)[:300]}", "tool_calls": [], "memo": None, "clarifying": False}

    # A memo ends the item: a later message starts a fresh intake (with the history for context)
    session["clarifications"] = session["clarifications"] + 1 if result["clarifying"] else 0
    session["turns"].append({"role": "assistant", "content": result["response"] if result["clarifying"] or not result["memo"] else result["memo"]["headline"]})

    return ChatResponse(response=result["response"] or "Sorry, no answer that time.", session_id=session_id,
                        tool_calls=result["tool_calls"], memo=result["memo"])


class DraftRequest(BaseModel):
    memo: dict  # the memo /chat returned


@app.post("/draft")
async def draft(request: DraftRequest):
    # An eBay-style title and description for the priced item. Stateless: the page sends the memo.
    return await research.draft_listing(request.memo)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    # Cloud Run sets PORT and needs 0.0.0.0; locally this serves http://localhost:8000
    uvicorn.run(app, host="0.0.0.0" if os.environ.get("PORT") else "127.0.0.1", port=int(os.environ.get("PORT", 8000)))
