import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage


from src.core.graph.state import (
    AgentState, 
    HumanFeedback, 
    AuditStatus, 
    WorkflowCommand
)
from src.core.config import settings


logger = structlog.get_logger(__name__)


class AuditorAgent:
    """
    Auditor agent for human-in-the-loop review and decision preparation.
    
    This agent prepares comprehensive audit reports and facilitates
    human review and decision-making processes.
    
    Features:
    - Comprehensive audit report generation
    - Human review workflow management
    - Decision recommendation system
    - Feedback processing and integration
    - Audit trail maintenance
    """
    
    def __init__(self):
        """Initialize auditor agent with LLM and configuration."""
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.DEFAULT_MODEL,
            temperature=0.1,  # Low temperature for consistent reporting
            max_tokens=settings.MAX_TOKENS
        )
        
        # Performance metrics
        self._metrics = {
            "total_reviews": 0,
            "avg_review_time": 0.0,
            "approval_rate": 0.0
        }
        
        logger.info("Initialized AuditorAgent")
    
    async def prepare_audit_report(self, state: AgentState) -> str:
        """
        Generate comprehensive audit report for human review.
        
        Args:
            state: Current audit state with all findings
        
        Returns:
            Formatted audit report
        """
        try:
            report_sections = []
            
            # 1. Executive Summary
            executive_summary = await self._generate_executive_summary(state)
            report_sections.append(executive_summary)
            
            # 2. Compliance Overview
            compliance_overview = await self._generate_compliance_overview(state)
            report_sections.append(compliance_overview)
            
            # 3. Detailed Findings
            detailed_findings = await self._generate_detailed_findings(state)
            report_sections.append(detailed_findings)
            
            # 4. Risk Assessment
            risk_assessment = await self._generate_risk_assessment(state)
            report_sections.append(risk_assessment)
            
            # 5. Recommendations
            recommendations = await self._generate_recommendations(state)
            report_sections.append(recommendations)
            
            # 6. Legal References
            legal_references = await self._generate_legal_references(state)
            report_sections.append(legal_references)
            
            # Combine all sections
            full_report = "\n\n".join(report_sections)
            
            logger.info("Audit report generated", audit_id=state["audit_id"])
            return full_report
            
        except Exception as e:
            logger.error("Failed to prepare audit report", error=str(e))
            return f"Error generating audit report: {str(e)}"
    
    async def _generate_executive_summary(self, state: AgentState) -> str:
        """Generate executive summary of the audit."""
        try:
            risk_assessment = state.get("risk_assessment", {})
            violations = state.get("violations", [])
            found_laws = state.get("found_laws", [])
            
            system_prompt = """
            You are a senior compliance auditor. Generate a concise executive summary
            of the audit findings for senior management.
            
            Include:
            1. Overall compliance status
            2. Key risk areas
            3. Critical violations (if any)
            4. Immediate actions required
            5. Business impact assessment
            
            Keep it professional, clear, and actionable. Maximum 300 words.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Audit Context:
                - Document: {state.get('document_metadata', {}).get('filename', 'Unknown')}
                - Compliance Domain: {state.get('compliance_domain', 'Unknown')}
                - Risk Score: {risk_assessment.get('overall_score', 'N/A')}/100
                - Compliance Level: {risk_assessment.get('level', 'Unknown')}
                - Violations Found: {len(violations)}
                - Regulations Reviewed: {len(found_laws)}
                
                Generate executive summary:
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            return f"## Executive Summary\n\n{response.content.strip()}"
            
        except Exception as e:
            logger.error("Failed to generate executive summary", error=str(e))
            return "## Executive Summary\n\nUnable to generate summary due to system error."
    
    async def _generate_compliance_overview(self, state: AgentState) -> str:
        """Generate compliance overview section."""
        try:
            risk_assessment = state.get("risk_assessment", {})
            
            overview = f"""
## Compliance Overview

**Overall Risk Score:** {risk_assessment.get('overall_score', 'N/A')}/100
**Compliance Level:** {risk_assessment.get('level', 'Unknown')}
**Total Violations:** {risk_assessment.get('violation_count', 0)}
**High-Risk Violations:** {risk_assessment.get('high_risk_violations', 0)}
**Compliance Percentage:** {risk_assessment.get('compliance_percentage', 0):.1f}%

### Status Assessment
{risk_assessment.get('assessment_summary', 'No assessment available')}
"""
            
            return overview.strip()
            
        except Exception as e:
            logger.error("Failed to generate compliance overview", error=str(e))
            return "## Compliance Overview\n\nUnable to generate overview due to system error."
    
    async def _generate_detailed_findings(self, state: AgentState) -> str:
        """Generate detailed findings section."""
        try:
            violations = state.get("violations", [])
            
            if not violations:
                return "## Detailed Findings\n\nNo violations detected. The document appears to be compliant with applicable regulations."
            
            findings = "## Detailed Findings\n\n"
            
            # Group violations by severity
            severity_groups = {"critical": [], "high": [], "medium": [], "low": []}
            for violation in violations:
                severity = violation.get("severity", "medium")
                severity_groups[severity].append(violation)
            
            for severity in ["critical", "high", "medium", "low"]:
                group_violations = severity_groups[severity]
                if group_violations:
                    findings += f"\n### {severity.upper()} Severity Violations ({len(group_violations)})\n\n"
                    
                    for i, violation in enumerate(group_violations, 1):
                        findings += f"**{i}. {violation.get('description', 'No description')}**\n"
                        findings += f"- **Risk Impact:** {violation.get('risk_impact', 'Not specified')}\n"
                        findings += f"- **Affected Clauses:** {', '.join(violation.get('affected_clauses', ['Not specified']))}\n"
                        findings += f"- **Confidence:** {violation.get('confidence_score', 0):.1%}\n"
                        findings += f"- **Remediation:** {violation.get('remediation_suggestion', 'No suggestion provided')}\n\n"
            
            return findings.strip()
            
        except Exception as e:
            logger.error("Failed to generate detailed findings", error=str(e))
            return "## Detailed Findings\n\nUnable to generate findings due to system error."
    
    async def _generate_risk_assessment(self, state: AgentState) -> str:
        """Generate risk assessment section."""
        try:
            risk_assessment = state.get("risk_assessment", {})
            violations = state.get("violations", [])
            
            assessment = f"""
## Risk Assessment

### Overall Risk Analysis
{risk_assessment.get('assessment_summary', 'No assessment available')}

### Violation Breakdown
"""
            
            # Count violations by severity
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for violation in violations:
                severity = violation.get("severity", "medium")
                severity_counts[severity] += 1
            
            for severity, count in severity_counts.items():
                if count > 0:
                    assessment += f"- **{severity.title()}:** {count} violations\n"
            
            # Risk mitigation recommendations
            assessment += "\n### Risk Mitigation Priorities\n"
            if severity_counts["critical"] > 0:
                assessment += "1. **IMMEDIATE:** Address all critical violations\n"
            if severity_counts["high"] > 0:
                assessment += "2. **URGENT:** Resolve high-risk violations within 30 days\n"
            if severity_counts["medium"] > 0:
                assessment += "3. **IMPORTANT:** Address medium-risk violations in next review cycle\n"
            if severity_counts["low"] > 0:
                assessment += "4. **RECOMMENDED:** Consider low-risk violations in future improvements\n"
            
            return assessment.strip()
            
        except Exception as e:
            logger.error("Failed to generate risk assessment", error=str(e))
            return "## Risk Assessment\n\nUnable to generate assessment due to system error."
    
    async def _generate_recommendations(self, state: AgentState) -> str:
        """Generate actionable recommendations."""
        try:
            violations = state.get("violations", [])
            compliance_domain = state.get("compliance_domain", "")
            
            system_prompt = f"""
            You are a compliance consultant specializing in {compliance_domain} regulations.
            
            Based on the audit findings, provide specific, actionable recommendations for:
            1. Immediate corrective actions
            2. Process improvements
            3. Compliance program enhancements
            4. Monitoring and reporting mechanisms
            5. Staff training requirements
            
            Make recommendations practical, prioritized, and measurable.
            """
            
            # Summarize violations for context
            violation_summary = "\n".join([
                f"- {v.get('severity', 'medium')}: {v.get('description', 'No description')}"
                for v in violations[:10]  # Limit to top 10
            ])
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Audit Findings:
                {violation_summary}
                
                Compliance Domain: {compliance_domain}
                
                Generate recommendations:
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            return f"## Recommendations\n\n{response.content.strip()}"
            
        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))
            return "## Recommendations\n\nUnable to generate recommendations due to system error."
    
    async def _generate_legal_references(self, state: AgentState) -> str:
        """Generate legal references section."""
        try:
            found_laws = state.get("found_laws", [])
            violations = state.get("violations", [])
            
            if not found_laws:
                return "## Legal References\n\nNo specific regulations were identified in this audit."
            
            references = "## Legal References\n\n"
            
            # Group laws by relevance to violations
            cited_law_ids = set()
            for violation in violations:
                cited_law_ids.update(violation.get("relevant_laws", []))
            
            referenced_laws = [law for law in found_laws if law.get("id") in cited_law_ids]
            other_laws = [law for law in found_laws if law.get("id") not in cited_law_ids]
            
            if referenced_laws:
                references += "### Directly Referenced Regulations\n\n"
                for law in referenced_laws:
                    references += f"**{law.get('title', 'Untitled')}**\n"
                    references += f"- **Source:** {law.get('source', 'Unknown')}\n"
                    references += f"- **Article:** {law.get('article_id', 'Not specified')}\n"
                    references += f"- **Relevance:** {law.get('relevance_score', 0):.1%}\n\n"
            
            if other_laws:
                references += "### Additional Relevant Regulations\n\n"
                for law in other_laws[:5]:  # Limit to top 5
                    references += f"**{law.get('title', 'Untitled')}**\n"
                    references += f"- **Source:** {law.get('source', 'Unknown')}\n"
                    references += f"- **Relevance:** {law.get('relevance_score', 0):.1%}\n\n"
            
            return references.strip()
            
        except Exception as e:
            logger.error("Failed to generate legal references", error=str(e))
            return "## Legal References\n\nUnable to generate references due to system error."
    
    async def generate_decision_recommendation(self, state: AgentState) -> str:
        """
        Generate recommendation for human reviewer decision.
        
        Args:
            state: Current audit state
        
        Returns:
            Decision recommendation with rationale
        """
        try:
            risk_assessment = state.get("risk_assessment", {})
            violations = state.get("violations", [])
            
            risk_score = risk_assessment.get("overall_score", 0)
            critical_violations = sum(1 for v in violations if v.get("severity") == "critical")
            high_violations = sum(1 for v in violations if v.get("severity") == "high")
            
            system_prompt = """
            You are a senior compliance officer providing a recommendation to the reviewing authority.
            
            Based on the audit results, provide a clear recommendation:
            - APPROVE: If compliant with minor issues
            - REJECT: If critical violations exist
            - REQUEST_REVISION: If significant issues need addressing
            
            Include clear rationale for your recommendation.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Audit Results:
                - Risk Score: {risk_score}/100
                - Critical Violations: {critical_violations}
                - High Risk Violations: {high_violations}
                - Total Violations: {len(violations)}
                
                Provide your recommendation:
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error("Failed to generate decision recommendation", error=str(e))
            return "Unable to generate recommendation due to system error."
    
    async def process_human_feedback(
        self, 
        state: AgentState, 
        feedback: HumanFeedback
    ) -> AgentState:
        """
        Process human feedback and update state accordingly.
        
        Args:
            state: Current audit state
            feedback: Human reviewer feedback
        
        Returns:
            Updated state
        """
        try:
            # Update state with feedback
            state["human_feedback"] = feedback
            state["last_updated"] = datetime.utcnow()
            
            # Process decision
            decision = feedback.get("decision", "approve")
            
            if decision == "approve":
                state["human_approved"] = True
                state["requires_revision"] = False
                state["status"] = AuditStatus.APPROVED
                
            elif decision == "reject":
                state["human_approved"] = False
                state["requires_revision"] = False
                state["status"] = AuditStatus.REJECTED
                
            elif decision == "request_revision":
                state["human_approved"] = False
                state["requires_revision"] = True
                state["status"] = AuditStatus.RESEARCHING  # Loop back to research
                
                # Add suggested changes to error log for processing
                suggested_changes = feedback.get("suggested_changes", [])
                if suggested_changes:
                    state["error_log"].append(f"Revision requested: {', '.join(suggested_changes)}")
            
            logger.info("Processed human feedback", 
                       audit_id=state["audit_id"],
                       decision=decision,
                       reviewer=feedback.get("reviewer_name", "Unknown"))
            
            return state
            
        except Exception as e:
            logger.error("Failed to process human feedback", error=str(e))
            state["error_log"].append(f"Feedback processing error: {str(e)}")
            return state
    
    def create_review_package(self, state: AgentState) -> Dict[str, Any]:
        """
        Create a complete review package for human reviewer.
        
        Args:
            state: Current audit state
        
        Returns:
            Complete review package
        """
        try:
            package = {
                "audit_id": state["audit_id"],
                "document_info": state.get("document_metadata", {}),
                "compliance_domain": state.get("compliance_domain", ""),
                "audit_timestamp": state.get("start_time", datetime.utcnow()).isoformat(),
                "risk_assessment": state.get("risk_assessment", {}),
                "violation_count": len(state.get("violations", [])),
                "regulations_reviewed": len(state.get("found_laws", [])),
                "status": state.get("status", "unknown"),
                "requires_review": True,
                "review_deadline": (datetime.utcnow().replace(hour=17, minute=0) + 
                                  timedelta(days=1)).isoformat() if datetime.utcnow().hour >= 12 
                                  else (datetime.utcnow().replace(hour=17, minute=0)).isoformat()
            }
            
            return package
            
        except Exception as e:
            logger.error("Failed to create review package", error=str(e))
            return {"error": str(e)}


async def auditor_node(state: AgentState) -> AgentState:
    """
    LangGraph node function for the auditor agent.
    
    This function prepares comprehensive audit reports and facilitates
    human-in-the-loop review processes.
    """
    try:
        logger.info("Starting auditor phase", audit_id=state["audit_id"])
        
        # Update state
        state["status"] = AuditStatus.HUMAN_REVIEW
        state["current_node"] = "auditor"
        state["last_updated"] = datetime.utcnow()
        
        # Initialize auditor agent
        auditor = AuditorAgent()
        
        # Generate comprehensive audit report
        audit_report = await auditor.prepare_audit_report(state)
        state["final_report"] = audit_report
        
        # Generate decision recommendation
        recommendation = await auditor.generate_decision_recommendation(state)
        
        # Create review package
        review_package = auditor.create_review_package(state)
        state["checkpoint_data"]["review_package"] = review_package
        state["checkpoint_data"]["recommendation"] = recommendation
        
        # Update performance metrics
        state["performance_metrics"]["auditor_phase"] = {
            "report_generated": True,
            "review_package_created": True,
            "completion_time": datetime.utcnow().isoformat()
        }
        
        logger.info("Auditor phase completed", 
                   audit_id=state["audit_id"],
                   report_length=len(audit_report))
        
        # This is where the workflow will pause for human review
        # The human will need to provide feedback to resume
        
        return state
        
    except Exception as e:
        logger.error("Auditor phase failed", audit_id=state.get("audit_id"), error=str(e))
        
        # Update state with error
        state["status"] = AuditStatus.ERROR
        state["error_log"].append(f"Auditor phase error: {str(e)}")
        state["last_updated"] = datetime.utcnow()
        
        return state