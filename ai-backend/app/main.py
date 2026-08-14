"""FastAPI + LangChain AI backend for the Next.js chatbot.

Exposes:
  GET  /health
  POST /rag/search              — RAG retrieval for AI SDK tools
  POST /v1/chat/completions     — OpenAI-compatible chat (custom models)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Literal

from dotenv import load_dotenv

# Load ai-backend/.env so os.getenv picks up MIXROUTE_API_KEY etc.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chains.chat import stream_chat_completion
from app.rag.diagnosis import DiagnosisRequest, build_diagnosis_support, generate_diagnosis_narrative
from app.rag.search import get_index_status, search_knowledge
from app.routers.cxr_classifier import router as cxr_router
from app.routers.legacy_predict import router as legacy_predict_router

API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app = FastAPI(title="Xray Healthcare AI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cxr_router)
app.include_router(legacy_predict_router)


def _check_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/rag/status")
def rag_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_api_key(authorization)
    return get_index_status()


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class RagChunk(BaseModel):
    id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagSearchResponse(BaseModel):
    query: str
    chunks: list[RagChunk]


@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(
    body: RagSearchRequest,
    authorization: str | None = Header(default=None),
) -> RagSearchResponse:
    _check_api_key(authorization)
    chunks = search_knowledge(query=body.query, top_k=body.top_k)
    return RagSearchResponse(query=body.query, chunks=chunks)


@app.post("/rag/diagnose")
def rag_diagnose(
    body: DiagnosisRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(authorization)
    return build_diagnosis_support(body)


@app.post("/rag/diagnose/stream")
async def rag_diagnose_stream(
    body: DiagnosisRequest,
    authorization: str | None = Header(default=None),
):
    """Stream an LLM-written clinical narrative grounded in the RAG assessment."""
    _check_api_key(authorization)

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            token_stream = await generate_diagnosis_narrative(body)
            async for token in token_stream:
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n".encode()
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "healthcare-rag"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = 0.2


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    _check_api_key(authorization)

    if not body.stream:
        text = ""
        async for token in stream_chat_completion(
            model=body.model,
            messages=[m.model_dump() for m in body.messages],
        ):
            text += token
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    async def event_stream() -> AsyncIterator[bytes]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async for token in stream_chat_completion(
            model=body.model,
            messages=[m.model_dump() for m in body.messages],
        ):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
