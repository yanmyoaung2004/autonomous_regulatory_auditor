import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

from src.core.graph.state import (
    AgentState, 
    AuditStatus, 
    WorkflowCommand,
    validate_state_transition
)
from src.core.config import settings
from src.agents.researcher import researcher_node
from src.agents.adversary import adversary_node
from src.agents.auditor import auditor_node


def create_initial_state(
    contract_text: str,
    document_metadata: Dict[str, Any],
    compliance_domain: str,
    audit_id: Optional[str] = None
) -> AgentState:
    """Create initial state for a new audit."""
    return AgentState(
        # === Input Data ===
        contract_text=contract_text,
        document_metadata=document_metadata,
        compliance_domain=compliance_domain,
        
        # === Workflow Control ===
        audit_id=audit_id or str(uuid.uuid4()),
        status=AuditStatus.PENDING,
        current_node="start",
        start_time=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        
        # === Research Phase ===
        extracted_keywords=[],
        found_laws=[],
        retrieval_queries=[],
        search_strategy="hybrid",
        
        # === Adversarial Phase ===
        violations=[],
        risk_assessment={},
        adversarial_analysis="",
        
        # === Human Review Phase ===
        human_feedback=None,
        human_approved=False,
        requires_revision=False,
        
        # === Output ===
        final_report="",
        executive_summary="",
        recommendations=[],
        
        # === System Metadata ===
        error_log=[],
        performance_metrics={},
        checkpoint_data={}
    )


def start_node(state: AgentState) -> AgentState:
    """Initialize the audit process and transition to research phase."""
    state["status"] = AuditStatus.RESEARCHING
    state["current_node"] = "researcher"
    state["last_updated"] = datetime.utcnow()
    state["performance_metrics"]["start_time"] = datetime.utcnow().isoformat()
    
    return state


def routing_function(state: AgentState) -> str:
    """Determine the next node based on current state and conditions."""
    current_status = state["status"]
    
    if current_status == AuditStatus.ERROR:
        return "error_handler"
    elif current_status == AuditStatus.RESEARCHING:
        return "researcher"
    elif current_status == AuditStatus.ADVERSARIAL_ANALYSIS:
        return "adversary"
    elif current_status == AuditStatus.HUMAN_REVIEW:
        return "auditor"
    elif state["human_approved"]:
        return "generate_report"
    elif state["requires_revision"]:
        return "researcher"  # Loop back for revision
    else:
        return "auditor"  # Default to human review


def error_handler(state: AgentState) -> AgentState:
    """Handle errors and determine recovery strategy."""
    error_messages = state.get("error_log", [])
    if error_messages:
        latest_error = error_messages[-1]
        state["performance_metrics"]["last_error"] = latest_error
        state["performance_metrics"]["error_timestamp"] = datetime.utcnow().isoformat()
    
    # Attempt recovery by restarting research phase
    state["status"] = AuditStatus.RESEARCHING
    state["current_node"] = "researcher"
    state["last_updated"] = datetime.utcnow()
    
    return state


def generate_report_node(state: AgentState) -> AgentState:
    """Generate final compliance report."""
    violations = state.get("violations", [])
    found_laws = state.get("found_laws", [])
    
    # Generate executive summary
    risk_score = state.get("risk_assessment", {}).get("overall_score", 0)
    compliance_level = state.get("risk_assessment", {}).get("level", "UNKNOWN")
    
    summary = f"""
    Audit Report for {state.get('document_metadata', {}).get('filename', 'Unknown Document')}
    
    Compliance Level: {compliance_level}
    Risk Score: {risk_score}/100
    Violations Found: {len(violations)}
    Regulations Reviewed: {len(found_laws)}
    
    Status: {"APPROVED" if state.get("human_approved") else "PENDING REVIEW"}
    """
    
    state["executive_summary"] = summary.strip()
    state["status"] = AuditStatus.APPROVED if state.get("human_approved") else AuditStatus.HUMAN_REVIEW
    state["current_node"] = "end"
    state["last_updated"] = datetime.utcnow()
    state["performance_metrics"]["completion_time"] = datetime.utcnow().isoformat()
    
    return state


def create_audit_workflow() -> StateGraph:
    """Create and configure the ARA audit workflow graph."""
    
    # Initialize the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("start", start_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("adversary", adversary_node)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("error_handler", error_handler)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("end", lambda state: END)
    
    # Set entry point
    workflow.set_entry_point("start")
    
    # Add conditional edges for routing
    workflow.add_conditional_edges(
        "start",
        routing_function,
        {
            "researcher": "researcher",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "researcher",
        lambda state: "adversary" if state.get("found_laws") else "error_handler",
        {
            "adversary": "adversary",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "adversary",
        lambda state: "auditor" if state.get("violations") is not None else "error_handler",
        {
            "auditor": "auditor",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "auditor",
        routing_function,
        {
            "generate_report": "generate_report",
            "researcher": "researcher",  # Revision loop
            "auditor": "auditor",        # Stay in review
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "error_handler",
        lambda state: "researcher" if state.get("status") == AuditStatus.RESEARCHING else "end",
        {
            "researcher": "researcher",
            "end": "end"
        }
    )
    
    workflow.add_edge("generate_report", "end")
    
    return workflow


def compile_workflow_with_checkpointer() -> Any:
    """Compile the workflow with SQLite checkpointer for persistence."""
    
    # Create workflow
    workflow = create_audit_workflow()
    
    # For now, use InMemorySaver to avoid SQLite issues
    # In production, you can switch back to SqliteSaver
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    
    # Compile with interrupt before auditor for human-in-the-loop
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["auditor"]  # Human review breakpoint
    )
    
    return app


# Global compiled workflow instance
audit_app = compile_workflow_with_checkpointer()


def run_audit(
    contract_text: str,
    document_metadata: Dict[str, Any],
    compliance_domain: str,
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Run a complete audit workflow.
    
    Args:
        contract_text: The contract/document text to audit
        document_metadata: Document metadata
        compliance_domain: Compliance domain (CBM, MOTC, etc.)
        thread_id: Thread ID for state persistence
        config: Additional runnable configuration
    
    Returns:
        Audit results and state information
    """
    
    # Create initial state
    initial_state = create_initial_state(
        contract_text=contract_text,
        document_metadata=document_metadata,
        compliance_domain=compliance_domain,
        audit_id=thread_id
    )
    
    # Configure runnable
    runnable_config = config or {}
    if thread_id:
        runnable_config["configurable"] = {"thread_id": thread_id}
    
    try:
        # Run the workflow
        result = audit_app.invoke(initial_state, runnable_config)
        
        return {
            "success": True,
            "audit_id": result.get("audit_id"),
            "status": result.get("status"),
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "audit_id": thread_id
        }


def resume_audit(
    thread_id: str,
    command: WorkflowCommand,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Resume a paused audit with human command.
    
    Args:
        thread_id: Thread ID of the paused audit
        command: Human command for workflow control
        config: Additional runnable configuration
    
    Returns:
        Updated audit results
    """
    
    runnable_config = config or {}
    runnable_config["configurable"] = {"thread_id": thread_id}
    
    try:
        # Resume with command
        result = audit_app.invoke(command, runnable_config)
        
        return {
            "success": True,
            "audit_id": result.get("audit_id"),
            "status": result.get("status"),
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "audit_id": thread_id
        }


def get_audit_state(thread_id: str) -> Optional[AgentState]:
    """Get current state of a paused audit."""
    try:
        # Get state snapshot
        snapshot = audit_app.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values if snapshot else None
    except Exception:
        return None


def list_active_audits() -> List[str]:
    """List all active audit thread IDs."""
    # This would require implementing a method to list all threads in the checkpointer
    # For now, return empty list as this depends on checkpointer implementation
    return []