"""LangChain chat orchestration with optional RAG augmentation."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.rag.search import search_knowledge


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _to_lc_messages(messages: list[dict[str, Any]]):
    converted = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _extract_text(msg.get("content"))
        if role == "system":
            converted.append(SystemMessage(content=text))
        elif role == "assistant":
            converted.append(AIMessage(content=text))
        else:
            converted.append(HumanMessage(content=text))
    return converted


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _extract_text(msg.get("content"))
    return ""


def _build_llm():
    """Prefer real OpenAI/Groq if keyed; otherwise a deterministic stub LLM."""
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model=os.getenv("LANGCHAIN_CHAT_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
        )

    if openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=openai_key,
            model=os.getenv("LANGCHAIN_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )

    # Zero-config local stub so the stack runs without cloud keys.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(
        responses=[
            "Based on retrieved knowledge: review the cited guidelines and correlate with clinical context. "
            "(Configure GROQ_API_KEY or OPENAI_API_KEY on the LangChain backend for live model replies.)"
        ]
    )


async def stream_chat_completion(
    *, model: str, messages: list[dict[str, Any]]
) -> AsyncIterator[str]:
    use_rag = model in {"healthcare-rag", "custom-rag"} or model.endswith("-rag")
    user_query = _last_user_text(messages)
    context = ""

    if use_rag and user_query:
        chunks = search_knowledge(user_query, top_k=3)
        context = "\n\n".join(
            f"[{c['source']}] {c['content']}" for c in chunks
        )

    system_extra = (
        "You are a TB and pneumonia clinical decision-support assistant. "
        "Use retrieved guideline context and supplied patient facts, but do not invent lab results, "
        "imaging findings, diagnoses, or treatment details. Frame outputs as decision support for "
        "clinician review, not as a final diagnosis. If the question is outside TB, pneumonia, "
        "chest X-ray interpretation, or this product's AI workflow, politely redirect. "
        "For diagnostic answers, use this structure when possible: likely assessment, TB category "
        "if supported, supporting evidence, uncertainty or missing tests, recommended next diagnostic "
        "steps, and source references. Cite sources in brackets like [document.pdf#page=12]. "
        "If retrieved context is insufficient, say so clearly.\n\n"
        f"Retrieved context:\n{context or '(none)'}"
    )

    lc_messages = _to_lc_messages(messages)
    # Prepend RAG system guidance for RAG models
    if use_rag:
        lc_messages = [SystemMessage(content=system_extra), *lc_messages]

    llm = _build_llm()

    # Stream tokens when the model supports it; FakeList falls back to one chunk.
    if hasattr(llm, "astream"):
        async for chunk in llm.astream(lc_messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text if isinstance(text, str) else _extract_text(text)
        return

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system}"), ("human", "{question}")]
    )
    chain = (
        RunnableLambda(
            lambda _: {
                "system": system_extra if use_rag else "You are a helpful assistant.",
                "question": user_query or "Hello",
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    result = await chain.ainvoke({})
    yield result
