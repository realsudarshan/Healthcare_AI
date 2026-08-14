# Xray Healthcare — Product & Technical Plan

**Product:** AI clinical assistant for doctors supporting **tuberculosis (TB)** and **pneumonia** diagnosis and care planning.  
**Stack:** Next.js (frontend) + FastAPI (AI backend) + custom vision models + RAG, with **Hugging Face** for model training, hosting, and ops.  
**Status:** Scaffold exists (chat UI, Groq LLMs, LangChain FastAPI stub + seed RAG). This plan defines the path to a production-ready doctor assistant.

---

## 1. Vision & Scope

### Who it is for
- Pulmonologists, radiologists, and general clinicians reviewing chest X-rays
- Not a patient-facing chatbot; not a replacement for clinical judgment

### What it does
1. **Vision assist** — Analyze chest radiographs for signs suggestive of TB and/or pneumonia; return structured findings with confidence and caveats
2. **RAG assist** — Ground answers in guidelines, protocols, and curated medical knowledge (WHO, national TB programs, pneumonia guidelines)
3. **Chat assist** — Conversational Q&A that can combine image findings + retrieved literature + differential considerations
4. **Care support** — Suggest workup next steps, monitoring points, and documentation drafts (always labeled as decision support)

### Explicit non-goals (v1)
- Autonomous diagnosis or prescription
- Guaranteeing treatment outcomes
- Replacing PACS / hospital EHR integration (phase later)
- Training on identifiable patient data without consent and governance

### Clinical safety framing
Every vision and RAG answer must include:
- “Decision support only — verify with clinical judgment and local protocols”
- Uncertainty / low-confidence flags when model or retrieval is weak
- Citations for guideline-backed claims
- Clear separation of **image findings** vs **clinical recommendations**

---

## 2. Current State (repo today)

| Layer | What exists | Gap |
|--------|-------------|-----|
| **Frontend** | Next.js App Router chat (AI SDK), auth, multimodal upload UI | No TB/pneumonia clinical UX; vision models not wired; prompts still generic |
| **LLM** | Groq models via AI SDK; `langchain/healthcare-rag` custom route | No clinical specialty prompts; no structured report output |
| **AI backend** | FastAPI in `ai-backend/` — `/health`, `/rag/search`, `/v1/chat/completions` | Seed keyword RAG only; no embeddings/vector DB; no vision endpoints |
| **Knowledge** | Small seed corpus (`ai-backend/app/rag/corpus.py`) with pneumonia notes | No TB corpus, no curated guideline ingestion pipeline |
| **Models** | No custom vision model | Need train/finetune + HF Hub + inference API |
| **HF ops** | Not set up | Spaces / Inference Endpoints / Datasets / model cards missing |

---

## 3. Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Next.js (Vercel)                                           │
│  Auth · Chat UI · X-ray upload · Structured report view     │
│  AI SDK tools: searchKnowledge, analyzeChestXray            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (OpenAI-compat + domain APIs)
┌──────────────────────────▼──────────────────────────────────┐
│  FastAPI AI Backend (ai-backend/)                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Chat/RAG    │  │ Vision       │  │ Orchestrator        │ │
│  │ chain       │  │ inference    │  │ (route intent,      │ │
│  │ + citations │  │ client       │  │  fuse vision+RAG)   │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼─────────────────────┼────────────┘
          │                │                     │
          ▼                ▼                     ▼
   Vector store      HF Inference          LLM (Groq /
   (embeddings)      Endpoint /            HF / local)
   + guideline       custom Space
   corpus            (TB/PNA vision)
```

### Responsibility split

| Concern | Owner |
|---------|--------|
| Auth, chat history, UI, streaming UX | Next.js |
| Retrieval, embeddings, vision calls, fusion, eval hooks | FastAPI |
| Model weights, training jobs, inference hosting | Hugging Face Hub / Endpoints / Spaces |
| Postgres chat/auth | Existing Neon Postgres |

---

## 4. Hugging Face Ops Strategy

Use HF as the **model + data + inference control plane** so training and serving stay portable.

### 4.1 Assets on Hub

| Asset | Purpose |
|--------|---------|
| `datasets/...-chest-tb-pna` | Curated/annotated chest X-ray labels (TB / pneumonia / normal / other) — governance-approved only |
| `models/...-chest-classifier` (or detector) | Custom vision model for TB & pneumonia signals |
| `models/...-embeddings` or off-the-shelf | Document embeddings for RAG |
| Model cards | Intended use, limitations, demographics, contraindication for sole diagnosis |
| Evaluation reports | Sensitivity/specificity / AUC by class; failure modes |

### 4.2 Training & iteration
- Prototype with strong public baselines (e.g. CheXpert/MIMIC-CXR–compatible encoders, ViT / ConvNeXt / DenseNet variants) via `transformers` + `datasets`
- Fine-tune for **multi-label** outputs: `tuberculosis`, `pneumonia`, `normal`, optional `other_abnormality`
- Track runs with HF / Weights & Biases (optional); version every checkpoint on Hub
- Prefer **privacy-safe** data: de-identified public sets first; hospital data only under IRB/BAA

### 4.3 Inference hosting (pick by stage)

| Stage | Hosting | Notes |
|--------|---------|--------|
| Dev | Local FastAPI + `transformers` / `pipeline` | Fast iteration |
| Staging | HF **Inference Endpoint** or GPU Space | Stable URL, autoscaling later |
| Prod | Dedicated Inference Endpoint (GPU) behind FastAPI | FastAPI never exposes HF tokens to browser |

FastAPI owns the HF API token; Next.js only talks to FastAPI.

### 4.4 Suggested env vars (backend)

```
HF_TOKEN=
HF_VISION_MODEL_ID=org/chest-tb-pna-v1
HF_VISION_ENDPOINT_URL=          # if using Inference Endpoints
HF_EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
RAG_VECTOR_BACKEND=chroma|faiss|pgvector
```

---

## 5. Vision Model Plan

### 5.1 Clinical task definition
**Input:** Chest X-ray image (PA/AP; DICOM→PNG/JPEG conversion later).  
**Output (structured JSON):**

```json
{
  "findings": {
    "tuberculosis": { "present": true, "confidence": 0.82, "regions": ["upper_lobes"] },
    "pneumonia": { "present": false, "confidence": 0.11, "regions": [] }
  },
  "quality": { "view": "PA", "adequate": true, "issues": [] },
  "differential": ["reactivation TB", "post-primary pattern"],
  "disclaimer": "Decision support only. Not a diagnosis."
}
```

### 5.2 Model approach (phased)

1. **Phase A — Classifier baseline**  
   Multi-label image classifier fine-tuned on public CXR datasets with TB/pneumonia labels where available; report AUC and calibration.

2. **Phase B — Localization (optional)**  
   Heatmaps / Grad-CAM or detection heads so doctors see *where* the model attended.

3. **Phase C — Multimodal LLM fusion**  
   Vision findings + RAG chunks → LLM generates a **structured radiology-style assist note** (findings / impression / suggested next steps).

### 5.3 FastAPI surface

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/vision/analyze` | Upload or URL image → structured findings |
| POST | `/v1/chat/completions` | Existing; extend to accept image parts + call vision |
| GET | `/vision/models` | Active model id + version for audit |

### 5.4 Frontend UX
- Dedicated “Chest X-ray assist” flow in chat (drag/drop CXR)
- Side panel: findings cards (TB / Pneumonia), confidence bars, heatmap overlay (phase B)
- One-click “Explain with guidelines” → triggers RAG with vision summary as query context
- Export: copy structured note for clinical documentation (doctor edits before use)

---

## 6. RAG Plan

### 6.1 Knowledge domains (priority order)
1. **TB** — WHO TB guidelines, national programs, drug-resistant TB basics, CXR patterns (primary vs post-primary)
2. **Pneumonia** — CAP/HAP guidelines, radiographic patterns, severity scores (as *references*, not auto-calculated without inputs)
3. **CXR technique & pitfalls** — quality, mimics, TB vs bacterial pneumonia differentials
4. **Product / safety** — how to use the assistant; what it must not do

### 6.2 Pipeline

```
Sources (PDF/MD) → chunk + metadata (title, org, year, section)
                → embed (HF embedding model)
                → vector store
                → /rag/search (hybrid: dense + keyword boost for drug names)
                → cite in chat / vision fusion
```

### 6.3 Implementation upgrades (from current seed)

| Step | Change |
|------|--------|
| 1 | Replace keyword corpus with real guideline markdown/PDF under `ai-backend/knowledge/` |
| 2 | Add HF embeddings + Chroma/FAISS (local) or pgvector (shared with Postgres) |
| 3 | Metadata filters: `disease=tb|pneumonia`, `doc_type=guideline` |
| 4 | Citation UI in Next.js from `searchKnowledge` tool results |
| 5 | Eval set: 50–100 clinical questions with expected sources |

### 6.4 Fusion pattern (vision + RAG)
When an image is present:
1. Run `/vision/analyze`
2. Build retrieval query from findings + user question  
   e.g. `"upper lobe cavitary pattern tuberculosis chest x-ray management"`
3. Retrieve top-k chunks
4. LLM answers with **Findings → Impression → Suggested considerations → Citations**

---

## 7. Frontend Plan (Next.js)

### Keep
- Auth.js, chat history, AI SDK streaming, multimodal input shell
- Tool: `searchKnowledge` → FastAPI `/rag/search`

### Add / change
- Clinical system prompt focused on TB & pneumonia decision support
- Tool: `analyzeChestXray` → FastAPI `/vision/analyze`
- Model entry: vision-capable custom model (e.g. `langchain/chest-vision-rag`) with `vision: true`
- UI: findings panel, citation list, strong disclaimers
- Attachment validation: image types, size limits, optional anonymization reminder

### Deploy
- Next.js on Vercel (existing pattern)
- FastAPI on GPU-capable host or container; point `LANGCHAIN_API_URL` to it

---

## 8. Backend Module Layout (target)

```
ai-backend/
  app/
    main.py                 # FastAPI routes
    chains/
      chat.py               # text + RAG chat
      fusion.py             # vision findings + RAG → LLM
    rag/
      corpus.py             # → deprecate for loader
      ingest.py             # load knowledge/
      search.py             # vector + hybrid search
      embeddings.py         # HF embeddings
    vision/
      schemas.py            # structured findings models
      infer.py              # HF endpoint / local pipeline
      preprocess.py         # resize, normalize, DICOM later
    eval/
      rag_eval.py
      vision_eval.py
  knowledge/                # curated guidelines (git-lfs or external)
  requirements.txt
```

---

## 9. Phased Delivery Roadmap

### Phase 0 — Align & harden scaffold (1 week)
- [ ] Lock product scope and clinical disclaimer copy
- [ ] Point prompts at TB/pneumonia (not generic “helpful assistant”)
- [ ] Document runbooks for Next.js + FastAPI locally
- [ ] Add this plan link from root `README.md`

### Phase 1 — Real RAG (1–2 weeks)
- [ ] Curate initial TB + pneumonia guideline corpus
- [ ] HF embeddings + vector store in FastAPI
- [ ] Upgrade `/rag/search` + citation metadata
- [ ] Eval: retrieval hit-rate on labeled questions
- [ ] Frontend citation display polish

### Phase 2 — Vision baseline (2–4 weeks)
- [ ] Dataset selection + labeling schema (HF Dataset)
- [ ] Train/fine-tune multi-label CXR model; publish model card
- [ ] `/vision/analyze` + local/HF inference
- [ ] Wire `analyzeChestXray` tool + findings UI
- [ ] Offline metrics report (AUC, calibration, failure cases)

### Phase 3 — Fusion assistant (1–2 weeks)
- [ ] Orchestrate vision → RAG → structured clinical note
- [ ] Stream fused answers through existing `/v1/chat/completions`
- [ ] Doctor feedback buttons (thumbs / “incorrect finding”) for improvement loop

### Phase 4 — HF production ops (1–2 weeks)
- [ ] Move vision to HF Inference Endpoint
- [ ] Versioned model promotions (staging → prod)
- [ ] Monitoring: latency, error rate, confidence distributions
- [ ] Security review: tokens, PII in logs, retention policy

### Phase 5 — Clinical readiness (ongoing)
- [ ] Clinical advisor review of prompts and outputs
- [ ] Bias / subgroup checks if demographics available
- [ ] DICOM ingest (optional), EHR hooks (later)
- [ ] Regulatory path assessment (SaMD / local rules) before hospital rollout

---

## 10. Security, Privacy & Compliance

- Images and chat may be **PHI** — encrypt in transit; minimize retention; redact logs
- Never send HF/Groq keys to the browser
- Opt-in storage of uploaded X-rays; default ephemeral analyze-only mode for demos
- Access control: authenticated clinicians only (existing Auth.js)
- Audit trail: model version + retrieval sources on every assist note
- Legal/medical: engage clinical and compliance stakeholders before production claims

---

## 11. Success Metrics

| Area | Metric |
|------|--------|
| Vision | AUC / sensitivity-specificity per class; calibration ECE |
| RAG | Recall@k of correct guideline chunk; citation accuracy |
| Product | Time-to-first-assist note; doctor “useful” rate |
| Ops | p95 latency vision & chat; endpoint uptime |
| Safety | % answers with disclaimer + citations; escalation on low confidence |

---

## 12. Immediate Next Actions

1. Confirm clinical advisor + allowed public datasets for TB/pneumonia CXR
2. Start **Phase 1** corpus: add TB + pneumonia guideline markdown under `ai-backend/knowledge/`
3. Scaffold `ai-backend/app/vision/` with stub `/vision/analyze` returning schema-compatible JSON
4. Update Next.js prompts and add `analyzeChestXray` tool stub
5. Create HF org/repo placeholders for dataset + model + model card template

---

## 13. Open Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Vector DB | Chroma local / FAISS / pgvector | Chroma for dev; pgvector if sharing Neon |
| Vision host | Local / HF Space / Inference Endpoint | Local → Endpoint for staging+ |
| LLM for fusion | Groq / HF / OpenAI | Keep Groq for chat; swap later if needed |
| Label taxonomy | Binary TB vs multi-pathology | Multi-label TB + pneumonia + normal |
| DICOM in v1 | Yes / No | No — PNG/JPEG first |

---

*Living document — update as phases complete. Architecture builds on the existing Next.js + `ai-backend` FastAPI split rather than replacing it.*
