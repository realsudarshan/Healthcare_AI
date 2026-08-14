# Healthcare Support AI Backend (LangChain + FastAPI)

OpenAI-compatible chat + RAG search for the Healthcare Support app.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/rag/search` | Retrieval for AI SDK `searchKnowledge` tool |
| POST | `/v1/chat/completions` | Custom models (`healthcare-rag`, etc.) |
| POST | `/predict` | Chest X-ray prediction used by the frontend upload flow |

## Run

```powershell
cd ai-backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8001
```

Optional: set `GROQ_API_KEY` or `OPENAI_API_KEY` in `.env` for live generations.

## Next.js wiring

In the chatbot `.env.local`:

```
LANGCHAIN_API_URL=http://localhost:8001
LANGCHAIN_API_KEY=
```

The same backend base URL is used for chat/RAG and X-ray prediction.
Select model **Healthcare RAG (LangChain)** in the UI, or ask questions that trigger the `searchKnowledge` tool.
