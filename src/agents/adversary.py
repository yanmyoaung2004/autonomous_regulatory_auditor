import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.graph.state import (
    AgentState, 
    Violation, 
    RiskAssessment, 
    AuditStatus, 
    ComplianceLevel,
    calculate_risk_score,
    determine_compliance_level
)
from src.core.config import settings


logger = structlog.get_logger(__name__)


class AdversaryAgent:
    """
    Advanced adversary agent for red-teaming compliance analysis.
    
    This agent operates as a "red team" to identify potential violations,
    compliance gaps, and regulatory risks in contracts and documents.
    
    Features:
    - Multi-perspective violation analysis
    - Risk scoring and categorization
    - Regulatory gap identification
    - Adversarial reasoning patterns
    - Comprehensive violation documentation
    """
    
    def __init__(self):
        """Initialize adversary agent with LLM and configuration."""
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.DEFAULT_MODEL,
            temperature=0.2,  # Slightly higher for creative adversarial thinking
            max_tokens=settings.MAX_TOKENS
        )
        
        # Violation severity weights
        self.severity_weights = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25
        }
        
        # Performance metrics
        self._metrics = {
            "total_analyses": 0,
            "avg_violations_found": 0.0,
            "avg_risk_score": 0.0
        }
        
        logger.info("Initialized AdversaryAgent")
    
    async def analyze_contract_violations(
        self, 
        contract_text: str, 
        found_laws: List[Dict[str, Any]],
        compliance_domain: str
    ) -> List[Violation]:
        """
        Perform comprehensive adversarial analysis to identify violations.
        
        Args:
            contract_text: Contract/document text
            found_laws: Retrieved relevant regulations
            compliance_domain: Compliance domain
        
        Returns:
            List of identified violations with detailed analysis
        """
        try:
            violations = []
            
            # 1. Direct regulatory violations
            direct_violations = await self._identify_direct_violations(
                contract_text, found_laws, compliance_domain
            )
            violations.extend(direct_violations)
            
            # 2. Compliance gap analysis
            gap_violations = await self._identify_compliance_gaps(
                contract_text, found_laws, compliance_domain
            )
            violations.extend(gap_violations)
            
            # 3. Risk-based violations
            risk_violations = await self._identify_risk_violations(
                contract_text, found_laws, compliance_domain
            )
            violations.extend(risk_violations)
            
            # 4. Procedural violations
            procedural_violations = await self._identify_procedural_violations(
                contract_text, found_laws, compliance_domain
            )
            violations.extend(procedural_violations)
            
            # Remove duplicates and merge similar violations
            unique_violations = self._deduplicate_violations(violations)
            
            logger.info("Adversarial analysis completed", 
                       violations_found=len(unique_violations),
                       analysis_type="comprehensive")
            
            return unique_violations
            
        except Exception as e:
            logger.error("Failed to analyze contract violations", error=str(e))
            return []
    
    async def _identify_direct_violations(
        self, 
        contract_text: str, 
        found_laws: List[Dict[str, Any]],
        compliance_domain: str
    ) -> List[Violation]:
        """Identify direct regulatory violations."""
        violations = []
        
        try:
            for law in found_laws[:5]:  # Analyze top 5 most relevant laws
                violation = await self._analyze_law_violation(
                    contract_text, law, compliance_domain
                )
                if violation:
                    violations.append(violation)
            
        except Exception as e:
            logger.error("Failed to identify direct violations", error=str(e))
        
        return violations
    
    async def _analyze_law_violation(
        self, 
        contract_text: str, 
        law: Dict[str, Any],
        compliance_domain: str
    ) -> Optional[Violation]:
        """Analyze if contract violates a specific law."""
        try:
            system_prompt = f"""
            You are a regulatory compliance expert specializing in {compliance_domain} law.
            
            Analyze the following contract text against the provided regulation to identify
            any violations, non-compliance issues, or regulatory breaches.
            
            For each violation found:
            1. Specify the severity (critical, high, medium, low)
            2. Describe the violation clearly
            3. Identify affected contract clauses
            4. Explain the regulatory requirements
            5. Suggest remediation steps
            6. Provide confidence score (0.0-1.0)
            
            If NO violations are found, respond with "NO_VIOLATIONS".
            
            Be thorough and adversarial in your analysis - look for any potential issues.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Regulation: {law['title']}
                Source: {law['source']}
                Content: {law['content']}
                
                Contract Text:
                {contract_text[:2000]}...
                
                Analysis:
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            analysis = response.content.strip()
            
            if "NO_VIOLATIONS" in analysis.upper():
                return None
            
            # Parse the analysis to extract violation details
            return self._parse_violation_analysis(analysis, law)
            
        except Exception as e:
            logger.error("Failed to analyze law violation", error=str(e))
            return None
    
    def _parse_violation_analysis(self, analysis: str, law: Dict[str, Any]) -> Optional[Violation]:
        """Parse LLM analysis to extract structured violation data."""
        try:
            # Simple parsing - in production, use more sophisticated NLP
            lines = analysis.split('\n')
            
            violation = Violation(
                id=str(uuid.uuid4()),
                severity="medium",  # Default
                description="",
                affected_clauses=[],
                relevant_laws=[law.get("id", "unknown")],
                risk_impact="",
                remediation_suggestion="",
                confidence_score=0.7  # Default
            )
            
            # Extract key information (simplified parsing)
            for line in lines:
                line_lower = line.lower()
                
                if "severity:" in line_lower:
                    for severity in ["critical", "high", "medium", "low"]:
                        if severity in line_lower:
                            violation["severity"] = severity
                            break
                
                if "description:" in line_lower or "violation:" in line_lower:
                    violation["description"] = line.split(":", 1)[-1].strip()
                
                if "confidence:" in line_lower:
                    try:
                        score = float(line.split(":")[-1].strip())
                        violation["confidence_score"] = min(1.0, max(0.0, score))
                    except ValueError:
                        pass
            
            # Ensure we have a description
            if not violation["description"]:
                violation["description"] = analysis[:200] + "..."
            
            return violation
            
        except Exception as e:
            logger.error("Failed to parse violation analysis", error=str(e))
            return None
    
    async def _identify_compliance_gaps(
        self, 
        contract_text: str, 
        found_laws: List[Dict[str, Any]],
        compliance_domain: str
    ) -> List[Violation]:
        """Identify compliance gaps (missing required elements)."""
        violations = []
        
        try:
            system_prompt = f"""
            You are a compliance expert for {compliance_domain} regulations.
            
            Analyze the contract for missing compliance elements, incomplete disclosures,
            or gaps that could lead to regulatory issues. Focus on what SHOULD be there
            but isn't.
            
            For each gap found:
            1. Rate severity (critical, high, medium, low)
            2. Describe the missing requirement
            3. Explain why it's required
            4. Suggest what should be added
            5. Provide confidence score
            
            If no gaps are found, respond with "NO_GAPS".
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Contract Text:
                {contract_text[:1500]}...
                
                Relevant Regulations Context:
                {', '.join([law['title'] for law in found_laws[:3]])}
                
                Compliance Gap Analysis:
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            analysis = response.content.strip()
            
            if "NO_GAPS" not in analysis.upper():
                # Parse gaps as violations
                gap_violation = Violation(
                    id=str(uuid.uuid4()),
                    severity="medium",
                    description="Compliance gaps identified: " + analysis[:300],
                    affected_clauses=["entire_document"],
                    relevant_laws=[law.get("id") for law in found_laws],
                    risk_impact="Potential regulatory non-compliance",
                    remediation_suggestion="Add missing compliance elements as identified",
                    confidence_score=0.6
                )
                violations.append(gap_violation)
            
        except Exception as e:
            logger.error("Failed to identify compliance gaps", error=str(e))
        
        return violations
    
    async def _identify_risk_violations(
        self, 
        contract_text: str, 
        found_laws: List[Dict[str, Any]],
        compliance_domain: str
    ) -> List[Violation]:
        """Identify risk-based violations (potential future issues)."""
        violations = []
        
        try:
            # Domain-specific risk patterns
            risk_patterns = self._get_domain_risk_patterns(compliance_domain)
            
            for pattern in risk_patterns:
                if pattern["keyword"].lower() in contract_text.lower():
                    violation = Violation(
                        id=str(uuid.uuid4()),
                        severity=pattern["severity"],
                        description=pattern["description"],
                        affected_clauses=[pattern["keyword"]],
                        relevant_laws=[],
                        risk_impact=pattern["risk_impact"],
                        remediation_suggestion=pattern["remediation"],
                        confidence_score=0.8
                    )
                    violations.append(violation)
            
        except Exception as e:
            logger.error("Failed to identify risk violations", error=str(e))
        
        return violations
    
    def _get_domain_risk_patterns(self, compliance_domain: str) -> List[Dict[str, Any]]:
        """Get domain-specific risk patterns."""
        patterns = {
            "CBM": [
                {
                    "keyword": "unlicensed",
                    "severity": "critical",
                    "description": "Unlicensed financial operations detected",
                    "risk_impact " : "Regulatory enforcement action",
                    "remediation": "Obtain proper CBM license"
                },
                {
                    "keyword": "cash settlement",
                    "severity": "high",
                    "description": "Cash-based settlements may violate foreign exchange regulations",
                    "risk_impact": "Transaction rejection or penalties",
                    "remediation": "Use proper banking channels"
                }
            ],
            "MOTC": [
                {
                    "keyword": "unregistered",
                    "severity": "high",
                    "description": "Unregistered transport operations",
                    "risk_impact": "Fines and operational suspension",
                    "remediation": "Register with MOTC"
                }
            ]
        }
        
        return patterns.get(compliance_domain, [])
    
    async def _identify_procedural_violations(
        self, 
        contract_text: str, 
        found_laws: List[Dict[str, Any]],
        compliance_domain: str
    ) -> List[Violation]:
        """Identify procedural violations (documentation, process issues)."""
        violations = []
        
        try:
            # Check for missing standard elements
            required_elements = ["signatures", "dates", "terms", "conditions"]
            missing_elements = []
            
            for element in required_elements:
                if element not in contract_text.lower():
                    missing_elements.append(element)
            
            if missing_elements:
                violation = Violation(
                    id=str(uuid.uuid4()),
                    severity="low",
                    description=f"Missing standard contract elements: {', '.join(missing_elements)}",
                    affected_clauses=["document_structure"],
                    relevant_laws=[],
                    risk_impact="Potential legal enforceability issues",
                    remediation_suggestion="Add missing standard contract elements",
                    confidence_score=0.9
                )
                violations.append(violation)
            
        except Exception as e:
            logger.error("Failed to identify procedural violations", error=str(e))
        
        return violations
    
    def _deduplicate_violations(self, violations: List[Violation]) -> List[Violation]:
        """Remove duplicate and merge similar violations."""
        unique_violations = {}
        
        for violation in violations:
            # Use description as key for deduplication
            key = violation["description"][:100].lower()
            
            if key not in unique_violations:
                unique_violations[key] = violation
            else:
                # Merge with existing violation
                existing = unique_violations[key]
                # Update severity to higher one
                severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                if severity_order[violation["severity"]] > severity_order[existing["severity"]]:
                    existing["severity"] = violation["severity"]
                
                # Merge relevant laws
                existing["relevant_laws"].extend(violation["relevant_laws"])
                existing["relevant_laws"] = list(set(existing["relevant_laws"]))
        
        return list(unique_violations.values())
    
    async def calculate_risk_assessment(self, violations: List[Violation]) -> RiskAssessment:
        """Calculate comprehensive risk assessment."""
        try:
            # Calculate overall risk score
            overall_score = calculate_risk_score(violations)
            
            # Determine compliance level
            compliance_level = determine_compliance_level(overall_score)
            
            # Count violations by severity
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for violation in violations:
                severity_counts[violation["severity"]] += 1
            
            # Calculate compliance percentage
            max_possible_score = 100
            compliance_percentage = max(0, 100 - overall_score)
            
            # Generate assessment summary
            summary = self._generate_assessment_summary(
                overall_score, compliance_level, severity_counts
            )
            
            risk_assessment = RiskAssessment(
                overall_score=overall_score,
                level=compliance_level,
                violation_count=len(violations),
                high_risk_violations=severity_counts["critical"] + severity_counts["high"],
                compliance_percentage=compliance_percentage,
                assessment_summary=summary
            )
            
            return risk_assessment
            
        except Exception as e:
            logger.error("Failed to calculate risk assessment", error=str(e))
            
            # Return default assessment
            return RiskAssessment(
                overall_score=50,
                level=ComplianceLevel.UNKNOWN,
                violation_count=len(violations),
                high_risk_violations=0,
                compliance_percentage=50.0,
                assessment_summary="Risk assessment calculation failed"
            )
    
    def _generate_assessment_summary(
        self, 
        risk_score: int, 
        compliance_level: ComplianceLevel,
        severity_counts: Dict[str, int]
    ) -> str:
        """Generate human-readable assessment summary."""
        total_violations = sum(severity_counts.values())
        
        summary = f"""
        Risk Assessment Summary:
        - Overall Risk Score: {risk_score}/100
        - Compliance Level: {compliance_level.value}
        - Total Violations: {total_violations}
        - Critical Violations: {severity_counts['critical']}
        - High Risk Violations: {severity_counts['high']}
        """
        
        if risk_score >= 80:
            summary += "\n- Status: HIGH RISK - Immediate attention required"
        elif risk_score >= 60:
            summary += "\n- Status: MEDIUM RISK - Review and remediation recommended"
        elif risk_score >= 30:
            summary += "\n- Status: LOW RISK - Minor issues to address"
        else:
            summary += "\n- Status: COMPLIANT - No significant issues detected"
        
        return summary.strip()


async def adversary_node(state: AgentState) -> AgentState:
    """
    LangGraph node function for the adversary agent.
    
    This function performs adversarial analysis to identify violations,
    compliance gaps, and regulatory risks.
    """
    try:
        logger.info("Starting adversarial analysis", audit_id=state["audit_id"])
        
        # Update state
        state["status"] = AuditStatus.ADVERSARIAL_ANALYSIS
        state["current_node"] = "adversary"
        state["last_updated"] = datetime.utcnow()
        
        # Initialize adversary agent
        adversary = AdversaryAgent()
        
        # Analyze contract for violations
        violations = await adversary.analyze_contract_violations(
            state["contract_text"],
            state["found_laws"],
            state["compliance_domain"]
        )
        
        # Calculate risk assessment
        risk_assessment = await adversary.calculate_risk_assessment(violations)
        
        # Generate adversarial analysis summary
        analysis_summary = f"""
        Adversarial Analysis Completed:
        - Total Violations Found: {len(violations)}
        - Overall Risk Score: {risk_assessment['overall_score']}/100
        - Compliance Level: {risk_assessment['level']}
        - High Risk Issues: {risk_assessment['high_risk_violations']}
        """
        
        # Update state with results
        state["violations"] = violations
        state["risk_assessment"] = risk_assessment
        state["adversarial_analysis"] = analysis_summary.strip()
        state["status"] = AuditStatus.HUMAN_REVIEW
        
        # Update performance metrics
        state["performance_metrics"]["adversary_phase"] = {
            "violations_found": len(violations),
            "risk_score": risk_assessment["overall_score"],
            "completion_time": datetime.utcnow().isoformat()
        }
        
        logger.info("Adversarial analysis completed", 
                   audit_id=state["audit_id"],
                   violations=len(violations),
                   risk_score=risk_assessment["overall_score"])
        
        return state
        
    except Exception as e:
        logger.error("Adversarial analysis failed", audit_id=state.get("audit_id"), error=str(e))
        
        # Update state with error
        state["status"] = AuditStatus.ERROR
        state["error_log"].append(f"Adversary phase error: {str(e)}")
        state["last_updated"] = datetime.utcnow()
        
        return state
