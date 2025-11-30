import base64
import json
import re

import pymupdf
import pytz
from langchain_core.messages import HumanMessage

from langGraph_states import ClaimState, FileDetails, FileType, FinalDecisionState
from llm import LLM

llm_instance = LLM()
openai_llm = llm_instance.llm
timezone = pytz.timezone("Asia/Kolkata")


def PDFtoImages_node(state: ClaimState):
    if not state.files:
        return []

    for file in state.files:
        pymupdf_doc = pymupdf.open(stream=file.content_as_bytes, filetype="pdf")
        page_images = []
        for page in pymupdf_doc:
            pix = page.get_pixmap()
            page_images.append(pix.tobytes())
            file.page_images = page_images

    return state


async def classify_pdf_files(state: ClaimState) -> ClaimState:
    assert state.files is not None, "State must have files to classify"

    for file in state.files:
        assert file.page_images is not None, (
            "File must have page images for classification"
        )

        page_images: list[bytes] = file.page_images
        if not page_images:
            continue

        content: list[dict[str, str | dict[str, str]]] = []
        for img_bytes in page_images:
            _img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_img_b64}",
                    },
                }
            )

        # STRICT JSON-ONLY PROMPT with schema
        prompt_text = """
        Analyze this medical document. RESPOND WITH ONLY VALID JSON matching this exact schema. No other text.

        REQUIRED JSON FORMAT:
        {
            "doctype": "bill" | "dischargesummary" | "idcard" | "pharmacybill" | "claimform",
            "structureddata": <structured data in form of valid JSON>,
            "confidence": 0.0-1.0
        }

        EXTRACTION RULES BY DOCTYPE:

        "bill": { "amount": float, "date": "YYYY-MM-DD", "provider": "str", "diagnosis_codes": ["str"] }
        "dischargesummary": { "diagnosis": "str", "admission_date": "YYYY-MM-DD", "discharge_date": "YYYY-MM-DD", "doctor": "str" }
        "idcard": { "member_id": "str", "policy_number": "str", "insurer": "str", "valid_from": "YYYY-MM-DD", "valid_to": "YYYY-MM-DD" }
        "pharmacybill": { "amount": float, "date": "YYYY-MM-DD", "patient_name": "str", "medicines": [{"name": "str", "qty": int, "price": float}] }
        "claimform": { "claim_number": "str", "service_dates": "str", "diagnosis_codes": ["str"], "procedure_codes": ["str"] }

        OUTPUT ONLY JSON. No explanations or greetings.
        """

        content.append({"type": "text", "text": prompt_text})

        response = await openai_llm.ainvoke([HumanMessage(content=content)])
        file.details_raw = response
        try:
            # Extract JSON from response (handle any extra text)
            json_match = re.search(r"\{.*\}", response, re.DOTALL | re.MULTILINE)
            if json_match:
                data = json.loads(json_match.group())
                file_details_obj = FileDetails(
                    doctype=data.get("doctype", "unknown"),
                    structured_data=data.get("structureddata", {}),
                    confidence=data.get("confidence", 0.0),
                )
                # storing 3 representations to facilitate debugging
                file.details_obj = file_details_obj
                file.details_json = data
            else:
                raise ValueError("LLM response does not contain valid JSON")

        except Exception as e:
            file.details_obj = FileDetails(
                doctype="unknown",
                structured_data={},
                confidence=0.0,
                error=True,
            )
            raise ValueError(f"Failed to parse JSON from LLM response: {e}")

    return state


async def cross_validate_node(state: ClaimState) -> ClaimState:
    files = state.files
    assert files is not None, (
        "State must have files for cross-validation. No files found."
    )

    # Gather structured details from all files
    extracted_details = []
    for f in files:
        details = getattr(f, "details_obj", None)
        extracted_details.append(
            {
                "structured_data": details.structured_data if details else {},
            }
        )

    # Custom prompt guiding the LLM to perform cross-validation
    prompt_text = (
        "You are validating an insurance claim package using extracted JSON details per file. "
        "Use strict rule-based checks and conservative assumptions. Return ONLY valid JSON.\n\n"
        "Validation goals: \n"
        "1) total_amount: Sum all monetary amounts from 'bill' and 'pharmacybill' doctype.\n"
        "2) Compare total_amount against claimed_amount from 'claimform' (if present). If total exceeds claimed_amount, flag an issue.\n"
        "3) member_id_match: Check if member_id in 'idcard' appears consistently in other files.\n"
        "4) required_docs_present: Ensure presence of at least 'idcard' and 'claimform'.\n"
        "5) Identify any inconsistencies, missing fields, date anomalies, or low-confidence extractions.\n\n"
        "Return JSON ONLY with schema: {\n"
        "  'valid': boolean,\n"
        "  'issues': [string],\n"
        "  'total_amount': number,\n"
        "  'member_id_match': boolean,\n"
        "  'required_docs_present': boolean\n"
        "}. No explanations outside JSON."
    )

    content = [
        {"type": "text", "text": prompt_text},
        {
            "type": "text",
            "text": json.dumps({"files": extracted_details}, ensure_ascii=False),
        },
    ]

    response = await openai_llm.ainvoke([HumanMessage(content=content)])

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL | re.MULTILINE)
        if not json_match:
            raise ValueError("LLM response does not contain valid JSON")

        data = json.loads(json_match.group())
        # Normalize and fallback defaults
        validation = {
            "valid": bool(data.get("valid", False)),
            "issues": data.get("issues", []) or [],
            "total_amount": float(data.get("total_amount", 0.0)),
            "member_id_match": bool(data.get("member_id_match", False)),
            "required_docs_present": bool(data.get("required_docs_present", False)),
        }
    except Exception as e:
        validation = {
            "valid": False,
            "issues": [f"Validation LLM error: {e}"],
            "total_amount": 0.0,
            "member_id_match": False,
            "required_docs_present": False,
        }

    state.validation_result = validation
    return state


def final_decision_node(state: ClaimState) -> ClaimState:
    validation = state.validation_result
    if not validation:
        state.final_decision = FinalDecisionState.MANUAL_REVIEW
        return state

    if validation.get("valid") is True:
        state.final_decision = FinalDecisionState.APPROVED
    else:
        issues = validation.get("issues", [])
        if len(issues) == 0:
            state.final_decision = FinalDecisionState.APPROVED
        else:
            state.final_decision = FinalDecisionState.REJECTED

    return state
