from dataclasses import dataclass
from enum import Enum


class FileType(Enum):
    """Enum to hold definitions for one of the file types:
    bill, discharge_summary, id_card, pharmacy_bill, claim_form, other
    """

    BILL = "bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    ID_CARD = "id_card"
    PHARMACY_BILL = "pharmacy_bill"
    CLAIM_FORM = "claim_form"
    OTHER = "other"


@dataclass
class FileDetails:
    doctype: str
    structured_data: dict
    confidence: float
    error: bool = False


@dataclass
class File:
    file_hash: str

    page_images: list[bytes] | None = None
    file_type: FileType | None = None
    content_as_bytes: bytes | None = None

    details_raw: str | None = None
    details_json: dict | None = None
    details_obj: FileDetails | None = None


class FinalDecisionState(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


@dataclass
class ClaimState:
    files: list[File] | None = None
    vision_analysis_results: dict | None = None
    cross_validation_results: dict | None = None
    final_decision: FinalDecisionState | None = None
    validation_result: dict = dict()
