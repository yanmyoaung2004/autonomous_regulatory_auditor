import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.graph.state import AgentState, RetrievedLaw, AuditStatus
from src.core.config import settings
from src.services.vector.qdrant_service import qdrant_service


logger = structlog.get_logger(__name__)


class ResearcherAgent:
    """
    Advanced researcher agent for legal compliance analysis.
    
    Features:
    - Keyword extraction using LLM
    - Hybrid search (semantic + keyword)
    - Query optimization for legal texts
    - Multi-domain compliance focus
    - Performance tracking
    """
    
    def __init__(self):
        """Initialize researcher agent with LLM and vector service."""
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        self.vector_service = qdrant_service
        
        # Performance metrics
        self._metrics = {
            "total_searches": 0,
            "avg_laws_found": 0.0,
            "avg_search_time": 0.0
        }
        
        logger.info("Initialized ResearcherAgent")
    
    async def extract_keywords(self, contract_text: str, compliance_domain: str) -> List[str]:
        """
        Extract relevant keywords and search queries from contract text.
        
        Args:
            contract_text: The contract/document text
            compliance_domain: Compliance domain (CBM, MOTC, etc.)
        
        Returns:
            List of extracted keywords and phrases
        """
        try:
            system_prompt = f"""
            You are a legal compliance expert specializing in {compliance_domain} regulations.
            
            Extract the most important keywords, phrases, and legal concepts from the following contract text
            that would be relevant for finding related regulations and compliance requirements.
            
            Focus on:
            1. Business activities and operations
            2. Financial transactions and amounts
            3. Regulatory terms and references
            4. Compliance obligations
            5. Risk-related terms
            6. Specific legal articles or directives mentioned
            
            Return as a comma-separated list of keywords and short phrases (2-4 words max each).
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Contract text:\n\n{contract_text[:2000]}...")
            ]
            
            response = await self.llm.ainvoke(messages)
            keywords_text = response.content.strip()
            
            # Parse keywords
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            # Add domain-specific keywords
            domain_keywords = self._get_domain_keywords(compliance_domain)
            keywords.extend(domain_keywords)
            
            # Remove duplicates and limit
            unique_keywords = list(set(keywords))[:20]
            
            logger.info("Extracted keywords", count=len(unique_keywords), domain=compliance_domain)
            return unique_keywords
            
        except Exception as e:
            logger.error("Failed to extract keywords", error=str(e))
            # Fallback to basic keyword extraction
            return self._fallback_keyword_extraction(contract_text)
    
    def _get_domain_keywords(self, compliance_domain: str) -> List[str]:
        """Get domain-specific keywords for enhanced search."""
        domain_keywords = {
            "CBM": [
                "central bank", "foreign exchange", "remittance", "money transfer",
                "financial institution", "banking license", "currency exchange",
                "international transfer", "compliance", "anti-money laundering"
            ],
            "MOTC": [
                "ministry of transport", "vehicle registration", "transport license",
                "logistics", "shipping", "import export", "customs", "transport permit"
            ],
            "Myanmar_Banking": [
                "banking law", "financial regulation", "deposit", "loan", "credit",
                "interest rate", "bank account", "transaction", "financial services"
            ]
        }
        
        return domain_keywords.get(compliance_domain, ["compliance", "regulation", "legal"])
    
    def _fallback_keyword_extraction(self, text: str) -> List[str]:
        """Fallback keyword extraction using basic text processing."""
        # Simple keyword extraction based on common legal/business terms
        legal_terms = [
            "agreement", "contract", "compliance", "regulation", "law", "legal",
            "obligation", "liability", "risk", "financial", "transaction",
            "payment", "transfer", "license", "permit", "authority", "government"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in legal_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:10]
    
    async def generate_search_queries(
        self, 
        keywords: List[str], 
        contract_text: str,
        compliance_domain: str
    ) -> List[str]:
        """
        Generate optimized search queries from keywords.
        
        Args:
            keywords: Extracted keywords
            contract_text: Original contract text
            compliance_domain: Compliance domain
        
        Returns:
            List of search queries
        """
        try:
            # Create multiple query strategies
            queries = []
            
            # 1. Individual keyword queries
            queries.extend(keywords[:5])
            
            # 2. Combined keyword queries
            if len(keywords) >= 2:
                combined_queries = [
                    f"{keywords[0]} {keywords[1]}",
                    f"{compliance_domain} regulation",
                    f"compliance {keywords[0]}"
                ]
                queries.extend(combined_queries)
            
            # 3. Context-aware queries using LLM
            context_query = await self._generate_context_query(
                keywords[:3], compliance_domain
            )
            if context_query:
                queries.append(context_query)
            
            # 4. Article/Directive specific queries
            article_queries = self._extract_article_references(contract_text)
            queries.extend(article_queries)
            
            # Remove duplicates and limit
            unique_queries = list(set(queries))[:10]
            
            logger.info("Generated search queries", count=len(unique_queries))
            return unique_queries
            
        except Exception as e:
            logger.error("Failed to generate search queries", error=str(e))
            return keywords[:5]  # Fallback to keywords
    
    async def _generate_context_query(self, keywords: List[str], compliance_domain: str) -> Optional[str]:
        """Generate a context-aware search query using LLM."""
        try:
            system_prompt = f"""
            Create a comprehensive search query for {compliance_domain} regulations
            using these keywords: {', '.join(keywords)}
            
            The query should be specific enough to find relevant legal articles but broad enough
            to capture related regulations. Focus on compliance requirements and legal obligations.
            
            Return only the search query (no explanation).
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate search query:")
            ]
            
            response = await self.llm.ainvoke(messages)
            query = response.content.strip()
            
            return query if len(query) > 10 else None
            
        except Exception as e:
            logger.error("Failed to generate context query", error=str(e))
            return None
    
    def _extract_article_references(self, text: str) -> List[str]:
        """Extract specific article or directive references from text."""
        patterns = [
            r'(?:Directive|Article|Section|Regulation|Rule)\s+\d+[/]?\d*',
            r'(?:CBM|MOTC)\s+Directive\s+\d+[/]?\d*',
            r'Law\s+No\.?\s*\d+[/]?\d*',
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))[:5]
    
    async def search_regulations(
        self, 
        queries: List[str], 
        compliance_domain: str
    ) -> List[RetrievedLaw]:
        """
        Search for relevant regulations using multiple queries.
        
        Args:
            queries: Search queries
            compliance_domain: Compliance domain
        
        Returns:
            List of retrieved laws
        """
        try:
            all_results = []
            
            # Perform hybrid search for each query
            for query in queries:
                results = await self.vector_service.hybrid_search(
                    query=query,
                    limit=settings.RETRIEVAL_TOP_K,
                    compliance_domain=compliance_domain
                )
                all_results.extend(results)
            
            # Remove duplicates based on document ID
            unique_results = {}
            for result in all_results:
                doc_id = result["id"]
                if doc_id not in unique_results or result["relevance_score"] > unique_results[doc_id]["relevance_score"]:
                    unique_results[doc_id] = result
            
            # Sort by relevance score and limit results
            sorted_results = sorted(
                unique_results.values(),
                key=lambda x: x["relevance_score"],
                reverse=True
            )
            
            final_results = sorted_results[:settings.RERANK_TOP_K]
            
            logger.info("Regulation search completed", 
                       queries_count=len(queries),
                       total_found=len(all_results),
                       unique_found=len(final_results))
            
            return final_results
            
        except Exception as e:
            logger.error("Failed to search regulations", error=str(e))
            return []
    
    async def analyze_relevance(
        self, 
        laws: List[RetrievedLaw], 
        contract_text: str
    ) -> List[RetrievedLaw]:
        """
        Analyze and score relevance of retrieved laws to the contract.
        
        Args:
            laws: Retrieved laws
            contract_text: Contract text
        
        Returns:
            Laws with updated relevance scores
        """
        try:
            if not laws:
                return []
            
            # Use LLM to analyze relevance for top laws
            top_laws = laws[:5]  # Analyze top 5 most relevant laws
            
            for law in top_laws:
                relevance_score = await self._calculate_law_relevance(
                    law, contract_text
                )
                law["relevance_score"] = relevance_score
            
            # Re-sort all laws by updated scores
            sorted_laws = sorted(laws, key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info("Relevance analysis completed", laws_analyzed=len(top_laws))
            return sorted_laws
            
        except Exception as e:
            logger.error("Failed to analyze relevance", error=str(e))
            return laws
    
    async def _calculate_law_relevance(
        self, 
        law: RetrievedLaw, 
        contract_text: str
    ) -> float:
        """Calculate relevance score for a specific law."""
        try:
            system_prompt = """
            You are a legal compliance expert. Rate the relevance of the following regulation
            to the contract text on a scale of 0.0 to 1.0.
            
            Consider:
            1. Direct applicability to contract activities
            2. Specific compliance requirements mentioned
            3. Risk level and potential violations
            4. Jurisdictional relevance
            
            Return only the numerical score (0.0-1.0).
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Regulation: {law['title']}
                Content: {law['content'][:500]}...
                
                Contract: {contract_text[:500]}...
                
                Relevance score (0.0-1.0):
                """)
            ]
            
            response = await self.llm.ainvoke(messages)
            score_text = response.content.strip()
            
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Ensure valid range
            except ValueError:
                return law.get("relevance_score", 0.5)  # Fallback to original score
                
        except Exception as e:
            logger.error("Failed to calculate law relevance", error=str(e))
            return law.get("relevance_score", 0.5)


async def researcher_node(state: AgentState) -> AgentState:
    """
    LangGraph node function for the researcher agent.
    
    This function performs comprehensive legal research using hybrid search
    and LLM-powered analysis to find relevant regulations.
    """
    try:
        logger.info("Starting research phase", audit_id=state["audit_id"])
        
        # Update state
        state["status"] = AuditStatus.RESEARCHING
        state["current_node"] = "researcher"
        state["last_updated"] = datetime.utcnow()
        
        # Initialize researcher agent
        researcher = ResearcherAgent()
        
        # Extract keywords from contract
        keywords = await researcher.extract_keywords(
            state["contract_text"],
            state["compliance_domain"]
        )
        state["extracted_keywords"] = keywords
        
        # Generate search queries
        queries = await researcher.generate_search_queries(
            keywords,
            state["contract_text"],
            state["compliance_domain"]
        )
        state["retrieval_queries"] = queries
        
        # Search for relevant regulations
        found_laws = await researcher.search_regulations(
            queries,
            state["compliance_domain"]
        )
        
        # Analyze relevance
        relevant_laws = await researcher.analyze_relevance(
            found_laws,
            state["contract_text"]
        )
        
        # Update state with results
        state["found_laws"] = relevant_laws
        state["status"] = AuditStatus.ADVERSARIAL_ANALYSIS
        
        # Update performance metrics
        state["performance_metrics"]["research_phase"] = {
            "keywords_extracted": len(keywords),
            "queries_generated": len(queries),
            "laws_found": len(relevant_laws),
            "completion_time": datetime.utcnow().isoformat()
        }
        
        logger.info("Research phase completed", 
                   audit_id=state["audit_id"],
                   laws_found=len(relevant_laws))
        
        return state
        
    except Exception as e:
        logger.error("Research phase failed", audit_id=state.get("audit_id"), error=str(e))
        
        # Update state with error
        state["status"] = AuditStatus.ERROR
        state["error_log"].append(f"Research phase error: {str(e)}")
        state["last_updated"] = datetime.utcnow()
        
        return state
