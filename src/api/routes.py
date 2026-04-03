import os
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from io import BytesIO

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog

from src.core.config import settings
from src.core.graph.state import AgentState, HumanFeedback, DocumentMetadata
from src.core.graph.workflow import run_audit, resume_audit, get_audit_state, list_active_audits
from src.services.vector.qdrant_service import qdrant_service


logger = structlog.get_logger(__name__)

router = APIRouter()

# Security
security = HTTPBearer(auto_error=False)


# Dependency for authentication
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In production, validate JWT token here
    # For now, just check if token exists
    if not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    
    return {"user_id": "demo_user", "permissions": ["audit:read", "audit:write"]}


# Pydantic models for API requests/responses
class AuditRequest(BaseModel):
    """Request model for starting a new audit."""
    contract_text: str = Field(..., min_length=100, description="Contract/document text to audit")
    compliance_domain: str = Field(..., description="Compliance domain (CBM, MOTC, Myanmar_Banking)")
    document_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    
    @validator('compliance_domain')
    def validate_domain(cls, v):
        allowed_domains = ['CBM', 'MOTC', 'Myanmar_Banking']
        if v not in allowed_domains:
            raise ValueError(f"Compliance domain must be one of: {allowed_domains}")
        return v


class AuditResponse(BaseModel):
    """Response model for audit initiation."""
    success: bool
    audit_id: str
    status: str
    message: str
    estimated_completion_time: Optional[str] = None


class HumanReviewRequest(BaseModel):
    """Request model for human review feedback."""
    decision: str = Field(..., description="Decision: approve, reject, or request_revision")
    reviewer_name: str = Field(..., description="Name of the reviewer")
    comments: str = Field(default="", description="Review comments")
    suggested_changes: Optional[List[str]] = Field(default=None, description="Suggested changes for revision")
    
    @validator('decision')
    def validate_decision(cls, v):
        allowed_decisions = ['approve', 'reject', 'request_revision']
        if v not in allowed_decisions:
            raise ValueError(f"Decision must be one of: {allowed_decisions}")
        return v


class AuditStatusResponse(BaseModel):
    """Response model for audit status."""
    audit_id: str
    status: str
    current_node: str
    start_time: str
    last_updated: str
    requires_human_review: bool
    risk_score: Optional[int] = None
    violation_count: Optional[int] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class AuditListResponse(BaseModel):
    """Response model for listing audits."""
    active_audits: List[str]
    total_count: int


# Helper functions
def validate_file_upload(file: UploadFile) -> bool:
    """Validate uploaded file."""
    # Check file size
    if file.size and file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    # Check file type
    if file.filename:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
    
    return True


def create_document_metadata(file: UploadFile = None, text_content: str = None) -> DocumentMetadata:
    """Create document metadata from uploaded file or text."""
    metadata = DocumentMetadata(
        filename=file.filename if file else "text_input",
        file_type=os.path.splitext(file.filename)[1].lower() if file else "text",
        upload_time=datetime.utcnow(),
        file_size_bytes=file.size if file else len(text_content.encode('utf-8')),
        page_count=None,  # Could be extracted from PDF
        language="en"  # Could be detected
    )
    
    return metadata


async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file."""
    # This is a placeholder - in production, use appropriate libraries:
    # - PDF: PyPDF2, pdfplumber, or LlamaParse
    # - DOCX: python-docx
    # - TXT: direct reading
    
    content = await file.read()
    
    if file.filename.endswith('.txt'):
        return content.decode('utf-8')
    elif file.filename.endswith('.pdf'):
        # Placeholder for PDF extraction
        return f"PDF content extraction not implemented. File size: {len(content)} bytes."
    elif file.filename.endswith('.docx'):
        # Placeholder for DOCX extraction
        return f"DOCX content extraction not implemented. File size: {len(content)} bytes."
    else:
        return content.decode('utf-8', errors='ignore')


# API Endpoints

@router.post("/audit/start", response_model=AuditResponse)
async def start_audit(
    request: AuditRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Start a new compliance audit.
    
    This endpoint initiates the multi-agent audit workflow including:
    - Research phase: Legal research and regulation retrieval
    - Adversarial phase: Violation detection and risk assessment
    - Human review: Approval/rejection workflow
    """
    try:
        logger.info("Starting new audit", 
                   user=current_user["user_id"],
                   domain=request.compliance_domain)
        
        # Create document metadata
        document_metadata = create_document_metadata(
            text_content=request.contract_text
        )
        
        # Override with provided metadata if any
        if request.document_metadata:
            document_metadata.update(request.document_metadata)
        
        # Generate unique audit ID
        audit_id = str(uuid.uuid4())
        
        # Run the audit workflow
        result = run_audit(
            contract_text=request.contract_text,
            document_metadata=document_metadata,
            compliance_domain=request.compliance_domain,
            thread_id=audit_id
        )
        
        if result["success"]:
            logger.info("Audit started successfully", audit_id=audit_id)
            
            return AuditResponse(
                success=True,
                audit_id=audit_id,
                status=result["status"],
                message="Audit initiated successfully",
                estimated_completion_time=(datetime.utcnow() + timedelta(minutes=30)).isoformat()
            )
        else:
            logger.error("Failed to start audit", audit_id=audit_id, error=result.get("error"))
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start audit: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error starting audit", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/audit/upload", response_model=AuditResponse)
async def start_audit_with_file(
    file: UploadFile = File(...),
    compliance_domain: str = Form(...),
    current_user: Dict = Depends(get_current_user)
):
    """
    Start a compliance audit by uploading a file.
    
    Supports PDF, DOCX, and TXT files.
    """
    try:
        logger.info("Starting audit with file upload", 
                   user=current_user["user_id"],
                   filename=file.filename,
                   domain=compliance_domain)
        
        # Validate file
        validate_file_upload(file)
        
        # Extract text from file
        contract_text = await extract_text_from_file(file)
        
        if len(contract_text) < 100:
            raise HTTPException(
                status_code=400,
                detail="Extracted text is too short for meaningful analysis (minimum 100 characters)"
            )
        
        # Create document metadata
        document_metadata = create_document_metadata(file=file)
        
        # Generate audit ID
        audit_id = str(uuid.uuid4())
        
        # Run audit
        result = run_audit(
            contract_text=contract_text,
            document_metadata=document_metadata,
            compliance_domain=compliance_domain,
            thread_id=audit_id
        )
        
        if result["success"]:
            logger.info("File-based audit started successfully", 
                       audit_id=audit_id, 
                       filename=file.filename)
            
            return AuditResponse(
                success=True,
                audit_id=audit_id,
                status=result["status"],
                message=f"Audit initiated for file: {file.filename}",
                estimated_completion_time=(datetime.utcnow() + timedelta(minutes=30)).isoformat()
            )
        else:
            logger.error("Failed to start file-based audit", 
                        audit_id=audit_id, 
                        error=result.get("error"))
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start audit: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in file upload audit", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audit/{audit_id}/status", response_model=AuditStatusResponse)
async def get_audit_status(
    audit_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the current status of an audit.
    
    Returns detailed information about the audit progress,
    including current phase, risk scores, and violations found.
    """
    try:
        logger.info("Getting audit status", audit_id=audit_id, user=current_user["user_id"])
        
        # Get audit state
        state = get_audit_state(audit_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Audit with ID {audit_id} not found"
            )
        
        # Extract relevant information
        risk_assessment = state.get("risk_assessment", {})
        violations = state.get("violations", [])
        performance_metrics = state.get("performance_metrics", {})
        
        return AuditStatusResponse(
            audit_id=audit_id,
            status=state.get("status", "unknown"),
            current_node=state.get("current_node", "unknown"),
            start_time=state.get("start_time", datetime.utcnow()).isoformat(),
            last_updated=state.get("last_updated", datetime.utcnow()).isoformat(),
            requires_human_review=state.get("status") == "human_review",
            risk_score=risk_assessment.get("overall_score"),
            violation_count=len(violations),
            performance_metrics=performance_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting audit status", audit_id=audit_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audit/{audit_id}/report")
async def get_audit_report(
    audit_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the full audit report.
    
    Returns the complete audit report including findings,
    risk assessment, and recommendations. Available after
    the auditor phase completes.
    """
    try:
        logger.info("Getting audit report", audit_id=audit_id, user=current_user["user_id"])
        
        # Get audit state
        state = get_audit_state(audit_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Audit with ID {audit_id} not found"
            )
        
        # Check if report is available
        final_report = state.get("final_report")
        if not final_report:
            raise HTTPException(
                status_code=400,
                detail="Audit report not yet available. Audit may still be in progress."
            )
        
        return {
            "audit_id": audit_id,
            "report": final_report,
            "executive_summary": state.get("executive_summary", ""),
            "risk_assessment": state.get("risk_assessment", {}),
            "violations": state.get("violations", []),
            "recommendations": state.get("recommendations", []),
            "generated_at": state.get("last_updated", datetime.utcnow()).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting audit report", audit_id=audit_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/audit/{audit_id}/review")
async def submit_human_review(
    audit_id: str,
    review_request: HumanReviewRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit human review feedback for an audit.
    
    This endpoint is used when an audit is waiting for human review.
    The reviewer can approve, reject, or request revisions.
    """
    try:
        logger.info("Submitting human review", 
                   audit_id=audit_id, 
                   user=current_user["user_id"],
                   decision=review_request.decision)
        
        # Get current audit state
        state = get_audit_state(audit_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Audit with ID {audit_id} not found"
            )
        
        # Check if audit is waiting for review
        if state.get("status") != "human_review":
            raise HTTPException(
                status_code=400,
                detail=f"Audit {audit_id} is not currently waiting for human review"
            )
        
        # Create human feedback object
        human_feedback = HumanFeedback(
            reviewer_id=current_user["user_id"],
            reviewer_name=review_request.reviewer_name,
            timestamp=datetime.utcnow(),
            decision=review_request.decision,
            comments=review_request.comments,
            suggested_changes=review_request.suggested_changes
        )
        
        # Create workflow command
        command = {
            "action": "resume",
            "human_feedback": human_feedback
        }
        
        # Resume audit with feedback
        result = resume_audit(
            thread_id=audit_id,
            command=command
        )
        
        if result["success"]:
            logger.info("Human review submitted successfully", 
                       audit_id=audit_id, 
                       decision=review_request.decision)
            
            return {
                "success": True,
                "message": f"Review submitted successfully. Decision: {review_request.decision}",
                "audit_status": result["status"]
            }
        else:
            logger.error("Failed to submit human review", 
                        audit_id=audit_id, 
                        error=result.get("error"))
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit review: {result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting human review", audit_id=audit_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audits/active", response_model=AuditListResponse)
async def list_active_audits_endpoint(
    current_user: Dict = Depends(get_current_user)
):
    """
    List all currently active audits.
    
    Returns a list of audit IDs that are currently in progress.
    """
    try:
        logger.info("Listing active audits", user=current_user["user_id"])
        
        active_audits = list_active_audits()
        
        return AuditListResponse(
            active_audits=active_audits,
            total_count=len(active_audits)
        )
        
    except Exception as e:
        logger.error("Error listing active audits", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/audit/{audit_id}")
async def cancel_audit(
    audit_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Cancel an ongoing audit.
    
    Note: This is a placeholder implementation. In production,
    you would need to implement proper cancellation logic.
    """
    try:
        logger.info("Cancelling audit", audit_id=audit_id, user=current_user["user_id"])
        
        # Get audit state
        state = get_audit_state(audit_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Audit with ID {audit_id} not found"
            )
        
        # Check if audit can be cancelled (not completed)
        status = state.get("status")
        if status in ["approved", "rejected"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel audit with status: {status}"
            )
        
        # Placeholder for cancellation logic
        # In production, you would:
        # 1. Mark the audit as cancelled in the database
        # 2. Stop any running processes
        # 3. Clean up resources
        
        logger.warning("Audit cancellation not fully implemented", audit_id=audit_id)
        
        return {
            "success": True,
            "message": f"Audit {audit_id} cancellation requested. Note: Full cancellation not implemented yet."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error cancelling audit", audit_id=audit_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audit/{audit_id}/review-package")
async def get_review_package(
    audit_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get the review package for human review.
    
    Returns a structured package containing all information
    needed for a human reviewer to make a decision.
    """
    try:
        logger.info("Getting review package", audit_id=audit_id, user=current_user["user_id"])
        
        # Get audit state
        state = get_audit_state(audit_id)
        
        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Audit with ID {audit_id} not found"
            )
        
        # Check if audit is waiting for review
        if state.get("status") != "human_review":
            raise HTTPException(
                status_code=400,
                detail=f"Audit {audit_id} is not currently waiting for human review"
            )
        
        # Get review package from checkpoint data
        review_package = state.get("checkpoint_data", {}).get("review_package", {})
        recommendation = state.get("checkpoint_data", {}).get("recommendation", "")
        
        return {
            "audit_id": audit_id,
            "review_package": review_package,
            "recommendation": recommendation,
            "final_report": state.get("final_report", ""),
            "risk_assessment": state.get("risk_assessment", {}),
            "violations": state.get("violations", []),
            "found_laws": state.get("found_laws", []),
            "submitted_at": state.get("last_updated", datetime.utcnow()).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting review package", audit_id=audit_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
