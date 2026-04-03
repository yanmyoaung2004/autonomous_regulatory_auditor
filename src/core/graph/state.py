from typing import Annotated, List, TypedDict, Optional, Dict, Any, Literal
from operator import add
from datetime import datetime
from enum import Enum


class AuditStatus(str, Enum):
    """Audit workflow status enumeration."""
    PENDING = "pending"
    RESEARCHING = "researching"
    ADVERSARIAL_ANALYSIS = "adversarial_analysis"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ERROR = "error"


class ComplianceLevel(str, Enum):
    """Compliance level enumeration."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


class DocumentMetadata(TypedDict):
    """Metadata for processed documents."""
    filename: str
    file_type: str
    upload_time: datetime
    file_size_bytes: int
    page_count: Optional[int]
    language: Optional[str]


class RetrievedLaw(TypedDict):
    """Retrieved legal regulation information."""
    id: str
    title: str
    source: str  # e.g., "CBM Directive 7/2024"
    content: str
    article_id: Optional[str]
    relevance_score: float
    retrieval_method: str  # "semantic", "keyword", "hybrid"
    metadata: Dict[str, Any]


class Violation(TypedDict):
    """Detected compliance violation."""
    id: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    affected_clauses: List[str]
    relevant_laws: List[str]  # Law IDs
    risk_impact: str
    remediation_suggestion: str
    confidence_score: float


class RiskAssessment(TypedDict):
    """Comprehensive risk assessment."""
    overall_score: int  # 0-100
    level: ComplianceLevel
    violation_count: int
    high_risk_violations: int
    compliance_percentage: float
    assessment_summary: str


class HumanFeedback(TypedDict):
    """Human reviewer feedback."""
    reviewer_id: str
    reviewer_name: str
    timestamp: datetime
    decision: Literal["approve", "reject", "request_revision"]
    comments: str
    suggested_changes: Optional[List[str]]


class AgentState(TypedDict):
    """Main workflow state for the ARA audit system."""
    
    # === Input Data ===
    contract_text: str
    document_metadata: DocumentMetadata
    compliance_domain: str  # e.g., "CBM", "MOTC", "Myanmar_Banking"
    
    # === Workflow Control ===
    audit_id: str
    status: AuditStatus
    current_node: str
    start_time: datetime
    last_updated: datetime
    
    # === Research Phase ===
    extracted_keywords: List[str]
    found_laws: Annotated[List[RetrievedLaw], add]
    retrieval_queries: List[str]
    search_strategy: str
    
    # === Adversarial Phase ===
    violations: Annotated[List[Violation], add]
    risk_assessment: RiskAssessment
    adversarial_analysis: str
    
    # === Human Review Phase ===
    human_feedback: Optional[HumanFeedback]
    human_approved: bool
    requires_revision: bool
    
    # === Output ===
    final_report: str
    executive_summary: str
    recommendations: List[str]
    
    # === System Metadata ===
    error_log: Annotated[List[str], add]
    performance_metrics: Dict[str, Any]
    checkpoint_data: Dict[str, Any]


class WorkflowCommand(TypedDict):
    """Command for workflow control."""
    action: Literal["resume", "restart", "approve", "reject", "request_revision"]
    target_node: Optional[str]
    parameters: Dict[str, Any]


# === State Validation Functions ===

def validate_state_transition(current: AuditStatus, next_status: AuditStatus) -> bool:
    """Validate workflow state transitions."""
    valid_transitions = {
        AuditStatus.PENDING: [AuditStatus.RESEARCHING, AuditStatus.ERROR],
        AuditStatus.RESEARCHING: [AuditStatus.ADVERSARIAL_ANALYSIS, AuditStatus.ERROR],
        AuditStatus.ADVERSARIAL_ANALYSIS: [AuditStatus.HUMAN_REVIEW, AuditStatus.ERROR],
        AuditStatus.HUMAN_REVIEW: [AuditStatus.APPROVED, AuditStatus.REJECTED, AuditStatus.RESEARCHING],
        AuditStatus.APPROVED: [],  # Terminal state
        AuditStatus.REJECTED: [],   # Terminal state
        AuditStatus.ERROR: [AuditStatus.RESEARCHING, AuditStatus.PENDING],  # Recovery states
    }
    return next_status in valid_transitions.get(current, [])


def calculate_risk_score(violations: List[Violation]) -> int:
    """Calculate overall risk score from violations."""
    if not violations:
        return 0
    
    severity_weights = {"low": 10, "medium": 25, "high": 50, "critical": 100}
    total_score = sum(
        severity_weights[violation["severity"]] * violation["confidence_score"]
        for violation in violations
    )
    
    # Normalize to 0-100 scale
    return min(100, int(total_score))


def determine_compliance_level(risk_score: int) -> ComplianceLevel:
    """Determine compliance level from risk score."""
    if risk_score == 0:
        return ComplianceLevel.COMPLIANT
    elif risk_score <= 30:
        return ComplianceLevel.PARTIALLY_COMPLIANT
    elif risk_score <= 70:
        return ComplianceLevel.PARTIALLY_COMPLIANT
    else:
        return ComplianceLevel.NON_COMPLIANT