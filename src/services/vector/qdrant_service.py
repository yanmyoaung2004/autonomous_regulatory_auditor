import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
import structlog

from src.core.config import settings
from src.core.graph.state import RetrievedLaw


logger = structlog.get_logger(__name__)


class QdrantService:
    """
    Advanced vector database service with hybrid search capabilities.
    
    Features:
    - Semantic search using sentence transformers
    - Keyword search using BM25
    - Hybrid search with Reciprocal Rank Fusion (RRF)
    - Multi-domain support (CBM, MOTC, etc.)
    - Performance monitoring and caching
    """
    
    def __init__(self):
        """Initialize Qdrant service with embedding model and client."""
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.WORKFLOW_TIMEOUT_SECONDS
        )
        self.collection_name = settings.VECTOR_COLLECTION
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_size = settings.VECTOR_SIZE
        
        # BM25 index for keyword search
        self._bm25_index = None
        self._documents_corpus = []
        
        # Performance metrics
        self._search_metrics = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "cache_hits": 0
        }
        
        logger.info("Initialized QdrantService", collection=self.collection_name)
    
    async def initialize_collection(self) -> bool:
        """
        Initialize Qdrant collection with proper configuration.
        
        Returns:
            True if collection was created/verified successfully
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )
            
            if not collection_exists:
                # Create collection with optimized configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=32,  # Connectivity parameter
                            ef_construct=200,  # Indexing accuracy
                        )
                    ),
                    sparse_vectors_config=models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True,  # For large datasets
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,  # Optimize for search speed
                        memmap_threshold=50000,  # Use memory mapping for large vectors
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=False,  # Memory efficient
                        )
                    )
                )
                logger.info("Created new collection", collection=self.collection_name)
            else:
                logger.info("Collection already exists", collection=self.collection_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize collection", error=str(e))
            return False
    
    def _encode_text(self, text: str) -> List[float]:
        """Encode text to vector embedding."""
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error("Failed to encode text", error=str(e))
            raise
    
    def _build_bm25_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build BM25 index for keyword search."""
        try:
            # Extract and tokenize document content
            tokenized_docs = []
            self._documents_corpus = documents
            
            for doc in documents:
                # Simple tokenization - can be enhanced with proper preprocessing
                tokens = doc["content"].lower().split()
                tokenized_docs.append(tokens)
            
            self._bm25_index = BM25Okapi(tokenized_docs)
            logger.info("Built BM25 index", document_count=len(documents))
            
        except Exception as e:
            logger.error("Failed to build BM25 index", error=str(e))
            self._bm25_index = None
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents in Qdrant with both dense and sparse vectors.
        
        Args:
            documents: List of document dictionaries with content, metadata, etc.
        
        Returns:
            True if indexing was successful
        """
        try:
            if not await self.initialize_collection():
                return False
            
            points = []
            
            for i, doc in enumerate(documents):
                # Generate dense vector
                dense_vector = self._encode_text(doc["content"])
                
                # Generate sparse vector (simple term frequency)
                tokens = doc["content"].lower().split()
                sparse_vector = models.SparseVector(
                    indices=list(range(len(tokens))),
                    values=[1.0] * len(tokens)  # Simple TF
                )
                
                point = PointStruct(
                    id=doc.get("id", str(uuid.uuid4())),
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector
                    },
                    payload={
                        "content": doc["content"],
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "article_id": doc.get("article_id", ""),
                        "compliance_domain": doc.get("compliance_domain", ""),
                        "metadata": doc.get("metadata", {}),
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upsert for performance
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            # Build BM25 index for keyword search
            self._build_bm25_index(documents)
            
            logger.info("Successfully indexed documents", count=len(documents))
            return True
            
        except Exception as e:
            logger.error("Failed to index documents", error=str(e))
            return False
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = None,
        compliance_domain: Optional[str] = None
    ) -> List[RetrievedLaw]:
        """
        Perform semantic search using dense vectors.
        
        Args:
            query: Search query
            limit: Number of results to return
            compliance_domain: Filter by compliance domain
        
        Returns:
            List of retrieved laws with relevance scores
        """
        limit = limit or settings.RETRIEVAL_TOP_K
        
        try:
            # Encode query
            query_vector = self._encode_text(query)
            
            # Build filter if compliance domain specified
            query_filter = None
            if compliance_domain:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="compliance_domain",
                            match=MatchValue(value=compliance_domain)
                        )
                    ]
                )
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedVector(
                    name="dense",
                    vector=query_vector
                ),
                query_filter=query_filter,
                limit=limit,
                score_threshold=0.3  # Minimum similarity threshold
            )
            
            # Convert to RetrievedLaw format
            results = []
            for hit in search_result:
                payload = hit.payload
                results.append(RetrievedLaw(
                    id=str(hit.id),
                    title=payload.get("title", ""),
                    source=payload.get("source", ""),
                    content=payload.get("content", ""),
                    article_id=payload.get("article_id"),
                    relevance_score=hit.score,
                    retrieval_method="semantic",
                    metadata=payload.get("metadata", {})
                ))
            
            return results
            
        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []
    
    async def keyword_search(
        self, 
        query: str, 
        limit: int = None,
        compliance_domain: Optional[str] = None
    ) -> List[RetrievedLaw]:
        """
        Perform keyword search using BM25.
        
        Args:
            query: Search query
            limit: Number of results to return
            compliance_domain: Filter by compliance domain
        
        Returns:
            List of retrieved laws with relevance scores
        """
        limit = limit or settings.RETRIEVAL_TOP_K
        
        try:
            if not self._bm25_index or not self._documents_corpus:
                logger.warning("BM25 index not available, falling back to sparse vector search")
                return await self._sparse_vector_search(query, limit, compliance_domain)
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self._bm25_index.get_scores(query_tokens)
            
            # Get top documents
            top_indices = sorted(
                range(len(bm25_scores)), 
                key=lambda i: bm25_scores[i], 
                reverse=True
            )[:limit]
            
            # Convert to RetrievedLaw format
            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include relevant documents
                    doc = self._documents_corpus[idx]
                    results.append(RetrievedLaw(
                        id=doc.get("id", str(idx)),
                        title=doc.get("title", ""),
                        source=doc.get("source", ""),
                        content=doc.get("content", ""),
                        article_id=doc.get("article_id"),
                        relevance_score=float(bm25_scores[idx]),
                        retrieval_method="keyword",
                        metadata=doc.get("metadata", {})
                    ))
            
            return results
            
        except Exception as e:
            logger.error("Keyword search failed", error=str(e))
            return await self._sparse_vector_search(query, limit, compliance_domain)
    
    async def _sparse_vector_search(
        self, 
        query: str, 
        limit: int,
        compliance_domain: Optional[str] = None
    ) -> List[RetrievedLaw]:
        """Fallback sparse vector search using Qdrant."""
        try:
            # Build sparse query
            tokens = query.lower().split()
            sparse_query = models.SparseVector(
                indices=list(range(len(tokens))),
                values=[1.0] * len(tokens)
            )
            
            # Build filter
            query_filter = None
            if compliance_domain:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="compliance_domain",
                            match=MatchValue(value=compliance_domain)
                        )
                    ]
                )
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=sparse_query
                ),
                query_filter=query_filter,
                limit=limit
            )
            
            # Convert to RetrievedLaw format
            results = []
            for hit in search_result:
                payload = hit.payload
                results.append(RetrievedLaw(
                    id=str(hit.id),
                    title=payload.get("title", ""),
                    source=payload.get("source", ""),
                    content=payload.get("content", ""),
                    article_id=payload.get("article_id"),
                    relevance_score=hit.score,
                    retrieval_method="sparse_vector",
                    metadata=payload.get("metadata", {})
                ))
            
            return results
            
        except Exception as e:
            logger.error("Sparse vector search failed", error=str(e))
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = None,
        compliance_domain: Optional[str] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[RetrievedLaw]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Uses Reciprocal Rank Fusion (RRF) to merge results from both search methods.
        
        Args:
            query: Search query
            limit: Number of results to return
            compliance_domain: Filter by compliance domain
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
        
        Returns:
            List of retrieved laws with combined relevance scores
        """
        limit = limit or settings.RERANK_TOP_K
        
        try:
            # Perform both searches concurrently
            semantic_results, keyword_results = await asyncio.gather(
                self.semantic_search(query, limit * 2, compliance_domain),
                self.keyword_search(query, limit * 2, compliance_domain)
            )
            
            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                semantic_results, 
                keyword_results,
                semantic_weight,
                keyword_weight
            )
            
            # Return top results
            return fused_results[:limit]
            
        except Exception as e:
            logger.error("Hybrid search failed", error=str(e))
            # Fallback to semantic search
            return await self.semantic_search(query, limit, compliance_domain)
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievedLaw],
        keyword_results: List[RetrievedLaw],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[RetrievedLaw]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = sum(weight_i / (k + rank_i))
        where k is a constant (typically 60)
        """
        k = 60  # RRF constant
        combined_scores = {}
        
        # Process semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = result["id"]
            rrf_score = semantic_weight / (k + rank + 1)
            combined_scores[doc_id] = {
                "result": result,
                "score": rrf_score,
                "retrieval_method": "hybrid"
            }
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result["id"]
            rrf_score = keyword_weight / (k + rank + 1)
            
            if doc_id in combined_scores:
                # Combine scores
                combined_scores[doc_id]["score"] += rrf_score
                combined_scores[doc_id]["result"]["relevance_score"] = combined_scores[doc_id]["score"]
            else:
                combined_scores[doc_id] = {
                    "result": result,
                    "score": rrf_score,
                    "retrieval_method": "hybrid"
                }
                combined_scores[doc_id]["result"]["relevance_score"] = rrf_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["result"] for item in sorted_results]
    
    async def get_document_by_id(self, doc_id: str) -> Optional[RetrievedLaw]:
        """Retrieve a specific document by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False
            )
            
            if result:
                payload = result[0].payload
                return RetrievedLaw(
                    id=str(result[0].id),
                    title=payload.get("title", ""),
                    source=payload.get("source", ""),
                    content=payload.get("content", ""),
                    article_id=payload.get("article_id"),
                    relevance_score=1.0,
                    retrieval_method="id_lookup",
                    metadata=payload.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error("Failed to retrieve document", doc_id=doc_id, error=str(e))
            return None
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the search service."""
        return self._search_metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Qdrant service."""
        try:
            # Test collection access
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            # Get collection info if exists
            collection_info = None
            if collection_exists:
                collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "status": "healthy" if collection_exists else "unhealthy",
                "collection_exists": collection_exists,
                "collection_info": collection_info,
                "search_metrics": self.get_search_metrics()
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "collection_exists": False
            }


# Global service instance
qdrant_service = QdrantService()
