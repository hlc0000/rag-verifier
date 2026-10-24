from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass
class EvidenceCard:
    content: str
    source: str = ""
    locator: str = ""
    heading_path: list[str] = field(default_factory=list)
    citability_score: float = 0.0
    cross_refs: list[str] = field(default_factory=list)
    completeness: str = "unknown"


@dataclass
class VerificationResult:
    status: VerificationStatus
    reason: str
    details: dict = field(default_factory=dict)
