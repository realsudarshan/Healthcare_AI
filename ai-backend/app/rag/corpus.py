"""Seed knowledge corpus for local RAG demos (healthcare / x-ray themed)."""

SEED_DOCUMENTS = [
    {
        "id": "xr-chest-01",
        "source": "guidelines/chest-xray-basics.md",
        "content": (
            "A standard PA chest X-ray is taken with the patient upright, facing the detector, "
            "X-ray beam entering from the back. Look systematically for airways, bones, cardiac "
            "silhouette, diaphragm, and lung fields. Compare both sides for asymmetry."
        ),
    },
    {
        "id": "xr-chest-02",
        "source": "guidelines/pneumonia-signs.md",
        "content": (
            "Pneumonia on chest radiograph often appears as focal or lobar consolidation with "
            "air bronchograms. Community-acquired pneumonia commonly affects lower lobes. "
            "Always correlate with clinical symptoms: fever, cough, and elevated WBC."
        ),
    },
    {
        "id": "xr-chest-03",
        "source": "guidelines/pneumothorax.md",
        "content": (
            "Pneumothorax shows a visceral pleural line with absent lung markings peripheral to it. "
            "In tension pneumothorax, look for mediastinal shift away from the affected side — "
            "this is a clinical emergency."
        ),
    },
    {
        "id": "ui-frontend-01",
        "source": "product/frontend-notes.md",
        "content": (
            "The Next.js chatbot uses Vercel AI SDK for streaming UI. Custom models and RAG "
            "should live in the LangChain FastAPI backend. Prefer tools like searchKnowledge "
            "for citations instead of stuffing large context into the system prompt."
        ),
    },
    {
        "id": "arch-01",
        "source": "product/architecture.md",
        "content": (
            "Recommended architecture: Next.js owns auth, chat UI, and streaming. LangChain "
            "backend owns retrieval, embeddings, custom model routing, and evaluation. "
            "Communicate via OpenAI-compatible /v1/chat/completions and /rag/search."
        ),
    },
]
