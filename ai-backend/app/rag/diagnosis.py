"""Clinical decision-support helpers for guideline-grounded TB/pneumonia RAG."""

from __future__ import annotations

import os
import re
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from app.rag.search import is_medical_scope, search_knowledge


class PatientContext(BaseModel):
    demographics: dict[str, Any] = Field(default_factory=dict)
    symptoms: dict[str, Any] = Field(default_factory=dict)
    duration: dict[str, Any] = Field(default_factory=dict)
    xray_findings: dict[str, Any] = Field(default_factory=dict)
    lab_tests: dict[str, Any] = Field(default_factory=dict)
    microbiology: dict[str, Any] = Field(default_factory=dict)
    tb_tests: dict[str, Any] = Field(default_factory=dict)
    risk_factors: dict[str, Any] = Field(default_factory=dict)
    previous_tb_history: dict[str, Any] = Field(default_factory=dict)
    treatment_history: dict[str, Any] = Field(default_factory=dict)
    comorbidities: dict[str, Any] = Field(default_factory=dict)
    model_predictions: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class DiagnosisRequest(BaseModel):
    query: str = Field(
        default="Use the collected TB type, lab results, X-ray result, and patient details to provide TB diagnosis and medicine-support information from NNF 2018."
    )
    patient: PatientContext
    top_k: int = Field(default=6, ge=1, le=12)


def _flatten(prefix: str, value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            lines.extend(_flatten(f"{prefix}.{key}" if prefix else str(key), child))
        return lines
    if isinstance(value, list):
        return [f"{prefix}: {', '.join(str(item) for item in value)}"]
    return [f"{prefix}: {value}"]


def patient_to_text(patient: PatientContext) -> str:
    data = patient.model_dump()
    lines: list[str] = []
    for key, value in data.items():
        lines.extend(_flatten(key, value))
    return "\n".join(lines) or "No structured patient facts supplied."


def _has_positive(patient_text: str, terms: list[str]) -> bool:
    lower = patient_text.lower()
    return any(term in lower for term in terms)


def _field_has_result(patient_text: str, field_terms: list[str], result_terms: list[str]) -> bool:
    lower = patient_text.lower()
    for field in field_terms:
        for result in result_terms:
            if re.search(rf"{re.escape(field)}[^\n]{{0,48}}\b{re.escape(result)}\b", lower):
                return True
            if re.search(rf"{re.escape(result)}[^\n]{{0,48}}\b{re.escape(field)}\b", lower):
                return True
    return False


def _field_has_completed_result(
    patient_text: str,
    field_terms: list[str],
    completed_terms: list[str] | None = None,
) -> bool:
    completed = completed_terms or ["detected", "not_detected", "not detected", "resistant", "susceptible"]
    return _field_has_result(patient_text, field_terms, completed)


def _infer_tb_category(patient_text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    lower = patient_text.lower()

    mtb_detected = _has_positive(
        lower,
        ["mtb detected", "gene xpert detected", "genexpert detected"],
    ) or _field_has_result(lower, ["mtb", "genexpert", "gene xpert"], ["detected"])
    rif_resistant = _has_positive(
        lower,
        ["rifampicin resistant", "rif resistant", "rpob", "rr-tb"],
    ) or _field_has_result(lower, ["rifampicin", "rif"], ["resistant"])
    inh_resistant = _has_positive(
        lower,
        ["isoniazid resistant", "inh resistant", "katg", "inha"],
    ) or _field_has_result(lower, ["isoniazid", "inh"], ["resistant"])
    fq_resistant = _has_positive(
        lower,
        ["fluoroquinolone resistant", "gyra", "gyrb"],
    ) or _field_has_result(lower, ["fluoroquinolone", "fq"], ["resistant"])
    group_a_resistant = _has_positive(
        lower,
        ["bedaquiline resistant", "linezolid resistant", "group a resistant"],
    ) or _field_has_result(
        lower,
        ["bedaquiline", "linezolid", "group a"],
        ["resistant"],
    )
    rif_susceptible = _has_positive(
        lower,
        ["rifampicin susceptible", "rif susceptible"],
    ) or _field_has_result(lower, ["rifampicin", "rif"], ["susceptible"])
    inh_susceptible = _has_positive(
        lower,
        ["isoniazid susceptible", "inh susceptible"],
    ) or _field_has_result(lower, ["isoniazid", "inh"], ["susceptible"])

    if not mtb_detected:
        return "TB not microbiologically confirmed from supplied data", [
            "No explicit MTB detected result was found in the patient context."
        ]

    reasons.append("MTB detection is present in the supplied patient facts.")
    if rif_resistant and fq_resistant and group_a_resistant:
        return "XDR-TB supported", reasons + [
            "Rifampicin, fluoroquinolone, and Group A drug resistance markers are present."
        ]
    if rif_resistant and fq_resistant:
        return "Pre-XDR-TB supported; XDR needs Group A DST", reasons + [
            "Rifampicin and fluoroquinolone resistance markers are present."
        ]
    if rif_resistant:
        return "MDR/RR-TB supported; fluoroquinolone DST needed", reasons + [
            "Rifampicin resistance is present."
        ]
    if rif_susceptible and inh_resistant:
        return "Isoniazid-resistant rifampicin-susceptible TB supported", reasons + [
            "Rifampicin susceptibility and isoniazid resistance are present."
        ]
    if rif_susceptible and inh_susceptible:
        return "Drug-sensitive TB supported", reasons + [
            "Rifampicin and isoniazid susceptibility are present."
        ]
    return "TB confirmed; resistance category incomplete", reasons + [
        "MTB is detected but resistance/susceptibility results are incomplete."
    ]


def _summarize_tb_diagnosis(patient_text: str) -> tuple[str, list[str]]:
    lower = patient_text.lower()
    tb_signals = [
        "tuberculosis",
        "ai_finding: tuberculosis",
        "mtb detected",
        "genexpert detected",
        "afb positive",
        "night sweat",
        "weight loss",
        "cavitation",
        "upper lobe",
        "chronic cough",
    ]
    tb_hits = [term for term in tb_signals if term in lower]

    if tb_hits:
        return "TB diagnosis information generated from collected patient data", [
            f"Collected TB-supporting signals: {', '.join(tb_hits)}."
        ]
    return "TB diagnosis information requires more clinical detail", [
        "No explicit TB-supporting symptom, X-ray, or microbiology signals were found in the supplied patient context."
    ]


def _missing_tests(patient_text: str) -> list[str]:
    lower = patient_text.lower()
    missing: list[str] = []

    has_mtb_result = _field_has_completed_result(
        lower,
        ["mtb", "mtbdetected", "genexpert", "gene xpert", "mtb/rif"],
        ["detected", "not_detected", "not detected"],
    )
    has_rif_result = _field_has_completed_result(
        lower,
        ["rifampicin", "rifampicinresistance", "rif"],
        ["resistant", "susceptible"],
    )
    if not (has_mtb_result and has_rif_result):
        missing.append("GeneXpert MTB/RIF")

    if not _field_has_completed_result(lower, ["afb", "smear"], ["positive", "negative", "detected", "not_detected", "not detected"]):
        missing.append("AFB smear microscopy")

    has_culture_dst = _field_has_completed_result(
        lower,
        ["culture", "dst", "grouparesistance", "group a"],
        ["resistant", "susceptible"],
    )
    if not has_culture_dst:
        missing.append("Culture and DST")

    has_isoniazid_dst = _field_has_completed_result(
        lower,
        ["lpa", "isoniazid", "isoniazidresistance", "inh", "katg", "inha"],
        ["resistant", "susceptible"],
    )
    if not has_isoniazid_dst:
        missing.append("First-line LPA or isoniazid DST")

    has_fq_dst = _field_has_completed_result(
        lower,
        ["fluoroquinolone", "fluoroquinoloneresistance", "gyra", "gyrb", "mtbdrsl"],
        ["resistant", "susceptible"],
    )
    if not has_fq_dst:
        missing.append("Second-line LPA for fluoroquinolone resistance")

    return missing


def _medicine_support(patient_text: str, tb_category: str) -> dict[str, Any]:
    lower = patient_text.lower()
    support: dict[str, Any] = {
        "title": "Medicine suggestions from NNF 2018",
        "disclaimer": (
            "Medication support is guideline-grounded decision support only. "
            "A qualified clinician must confirm the diagnosis, allergy history, pregnancy status, "
            "renal/hepatic function, drug interactions, local protocol, and exact dose before prescribing."
        ),
        "recommendations": [],
        "cautions": [],
    }

    category = tb_category.lower()
    if "drug-sensitive tb" in category:
        support["recommendations"].extend(
            [
                {
                    "condition": "Drug-sensitive TB supported",
                    "medicine": "Isoniazid",
                    "suggestion": "NNF lists isoniazid for tuberculosis in combination with other drugs; adult dose 300 mg daily and child dose 10 mg/kg daily, maximum 300 mg daily.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=26",
                },
                {
                    "condition": "Drug-sensitive TB supported",
                    "medicine": "Rifampin/Rifampicin",
                    "suggestion": "NNF lists rifampin for short-course chemotherapy of tuberculosis in combination with other drugs; adult >50 kg dose 600 mg daily for category-I 6 months and category-II 8 months, child dose 15 mg/kg daily for the same durations.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=27",
                },
                {
                    "condition": "Drug-sensitive TB supported",
                    "medicine": "Pyrazinamide",
                    "suggestion": "NNF lists pyrazinamide for tuberculosis; adult <50 kg dose 1.5 g/day, adult >50 kg dose 2 g/day, child dose 25-35 mg/kg daily.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=26",
                },
                {
                    "condition": "Drug-sensitive TB supported",
                    "medicine": "Ethambutol",
                    "suggestion": "NNF lists ethambutol for tuberculosis in combination with other drugs; dose range 15-25 mg/kg daily.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=25",
                },
            ]
        )

    if "mdr" in category or "xdr" in category or "resistance" in category:
        support["recommendations"].extend(
            [
                {
                    "condition": "Drug-resistant TB concern",
                    "medicine": "Bedaquiline",
                    "suggestion": "NNF lists bedaquiline as part of a combination regimen for pulmonary MDR-TB when other effective regimens are not available; adult schedule starts 400 mg once daily for 2 weeks, then 200 mg three times weekly for 22 weeks.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=28",
                },
                {
                    "condition": "Drug-resistant TB concern",
                    "medicine": "Specialized second-line MDR-TB medicines",
                    "suggestion": "NNF essential-medicines listing includes second-line MDR-TB medicines such as bedaquiline, capreomycin, clofazimine, cycloserine, ethionamide, kanamycin, levofloxacin, linezolid, moxifloxacin, ofloxacin, and PAS, to be made available only in specialized centres following standard treatment protocol.",
                    "source": "TB_Pneumonia_Extract_NNF2018 (1).pdf#page=35",
                },
            ]
        )

    if "pregnancy_status: true" in lower:
        support["cautions"].append(
            "Pregnancy is marked present. Several TB and antibacterial monographs include pregnancy precautions; require clinician/specialist review before medicine selection."
        )
    if "hepatic_impairment: true" in lower or "hepatic" in lower:
        support["cautions"].append(
            "Hepatic impairment is marked or mentioned. Isoniazid, rifampin, pyrazinamide, and amoxicillin/clavulanate all require careful hepatic risk review."
        )
    if "renal_impairment: true" in lower or "renal" in lower:
        support["cautions"].append(
            "Renal impairment is marked or mentioned. Dose adjustment and renal monitoring may be needed, especially for ethambutol and injectable/second-line medicines."
        )
    if "diabetes_positive" in lower or "diabetes_status: diabetes_positive" in lower:
        support["cautions"].append(
            "Diabetes is present. Monitor infection response and glucose control; several NNF monographs list diabetes-related precautions."
        )
    if "grouparesistance: resistant" in lower or "group a" in lower and "resistant" in lower:
        support["cautions"].append(
            "Group A resistance was entered. Do not treat this as simple drug-sensitive TB without specialist DST review, even if rifampicin and isoniazid are susceptible."
        )

    if not support["recommendations"]:
        support["cautions"].append(
            "No medicine suggestion was generated because the available facts do not support TB or pneumonia treatment selection."
        )

    return support


def _nnf_medicine_chunks(patient_text: str, tb_category: str, top_k: int) -> list[dict]:
    queries = [
        "NNF 2018 isoniazid rifampin pyrazinamide ethambutol tuberculosis dosage schedule",
    ]
    category = tb_category.lower()
    lower = patient_text.lower()
    if "mdr" in category or "xdr" in category or "resistance" in category:
        queries.append(
            "NNF 2018 bedaquiline capreomycin cycloserine ethionamide linezolid moxifloxacin MDR-TB second-line medicines"
        )
    if "ai_finding: pneumonia" in lower or "consolidation" in lower or "lrti" in lower:
        queries.append(
            "NNF 2018 amoxicillin amoxicillin clavulanate moxifloxacin pneumonia lower respiratory tract infection dosage"
        )

    merged: dict[str, dict] = {}
    for query in queries:
        for chunk in search_knowledge(query, top_k=top_k):
            if chunk["source"].startswith("TB_Pneumonia_Extract_NNF2018"):
                merged[chunk["id"]] = chunk
    return list(merged.values())[:top_k]


def build_diagnosis_support(body: DiagnosisRequest) -> dict[str, Any]:
    patient_text = patient_to_text(body.patient)
    tb_category, tb_reasons = _infer_tb_category(patient_text)
    retrieval_query = (
        "NNF 2018 tuberculosis medicines diagnosis support "
        f"{tb_category} isoniazid rifampin rifampicin pyrazinamide ethambutol "
        "bedaquiline mdr-tb second-line medicines cautions\n\n"
        f"Patient facts:\n{patient_text}"
    )

    if not is_medical_scope(retrieval_query):
        return {
            "in_scope": False,
            "message": "This assistant only supports TB/pneumonia diagnostic decision-support questions.",
            "chunks": [],
        }

    chunks = search_knowledge(retrieval_query, top_k=body.top_k)
    nnf_chunks = _nnf_medicine_chunks(patient_text, tb_category, top_k=body.top_k)
    seen_chunk_ids = {chunk["id"] for chunk in nnf_chunks}
    chunks = [
        *nnf_chunks,
        *[chunk for chunk in chunks if chunk["id"] not in seen_chunk_ids],
    ][: body.top_k]
    tb_summary, diagnosis_reasons = _summarize_tb_diagnosis(patient_text)
    missing = _missing_tests(patient_text)
    medicine_support = _medicine_support(patient_text, tb_category)

    source_refs = []
    seen_sources: set[str] = set()
    for chunk in chunks:
        if chunk["source"] in seen_sources:
            continue
        seen_sources.add(chunk["source"])
        page = chunk.get("metadata", {}).get("page")
        suffix = f", page {page}" if page else ""
        source_refs.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "label": f"{chunk['source']}{suffix}",
            }
        )

    return {
        "in_scope": True,
        "assessment": {
            "diagnosis_summary": tb_summary,
            "most_likely": tb_summary,
            "tb_category": tb_category,
            "supporting_evidence": diagnosis_reasons + tb_reasons,
            "missing_or_uncertain": missing,
            "medicine_support": medicine_support,
            "safety_note": (
                "Decision support only. Confirm diagnosis, urgency, and treatment "
                "with a qualified clinician and the original lab reports."
            ),
        },
        "chunks": chunks,
        "patient_facts": patient_text,
        "source_references": source_refs,
    }


def build_generation_context(body: DiagnosisRequest) -> str:
    support = build_diagnosis_support(body)
    if not support.get("in_scope"):
        return str(support["message"])

    chunk_lines = [
        f"[{chunk['source']}] {re.sub(r'\\s+', ' ', chunk['content']).strip()}"
        for chunk in support["chunks"]
    ]
    return "\n\n".join(
        [
            "Patient facts:",
            support["patient_facts"],
            "Structured assessment:",
            json_like(support["assessment"]),
            "Retrieved guideline context:",
            "\n\n".join(chunk_lines),
        ]
    )


def json_like(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, indent=2)


_NARRATIVE_SYSTEM = (
    "You are a clinical decision-support assistant for TB and pneumonia. "
    "You are given structured patient facts, a rule-based assessment, and retrieved "
    "guideline excerpts from the Nepal TB Management Guidelines 2019 and NNF 2018. "
    "Write a concise, clinician-facing diagnostic narrative (200-350 words) that:\n"
    "1. States the most likely diagnosis and TB category if applicable.\n"
    "2. Lists the key supporting evidence from the patient facts.\n"
    "3. Flags missing or uncertain tests needed to confirm.\n"
    "4. Summarises the medicine suggestions with doses where available, citing the guideline source.\n"
    "5. Ends with a safety note reminding the clinician this is decision support only.\n"
    "Format with short paragraphs. Cite sources in brackets like [NNF2018#page=26]. "
    "Do not invent lab results, imaging findings, or diagnoses beyond what the facts support."
)


async def generate_diagnosis_narrative(body: DiagnosisRequest) -> AsyncIterator[str]:
    """Stream an LLM-written clinical narrative grounded in the RAG assessment."""
    context = build_generation_context(body)

    if not context or context.startswith("This assistant only"):
        async def _out_of_scope() -> AsyncIterator[str]:
            yield context or "This query is outside the TB/pneumonia RAG scope."
        return _out_of_scope()

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_NARRATIVE_SYSTEM),
        HumanMessage(content=context),
    ]

    mixroute_key = os.getenv("MIXROUTE_API_KEY")
    mixroute_url = os.getenv("MIXROUTE_BASE_URL", "https://api.mixroute.ai/v1")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if mixroute_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=mixroute_key,
            base_url=mixroute_url,
            model=os.getenv("LANGCHAIN_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            streaming=True,
        )
    elif groq_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model=os.getenv("LANGCHAIN_CHAT_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            streaming=True,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=openai_key,
            model=os.getenv("LANGCHAIN_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            streaming=True,
        )
    else:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        llm = FakeListChatModel(
            responses=[
                "Based on the retrieved guidelines and patient facts, no LLM key is configured. "
                "Set MIXROUTE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY in ai-backend/.env for live narrative generation."
            ]
        )

    async def _stream() -> AsyncIterator[str]:
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text and isinstance(text, str):
                yield text

    return _stream()
