"""Retrieval Agent - Handles query retrieval and reranking."""
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from sentence_transformers import CrossEncoder

from ..config import settings
from ..schemas import QueryRequest, Modality
from ..logging_config import get_logger

logger = get_logger(__name__)


class RetrievalAgent:
    """Agent for document retrieval."""
    
    def __init__(self):
        """Initialize retrieval agent."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Initialize reranker
        try:
            self.reranker = CrossEncoder(settings.RERANK_MODEL)
            self.reranker_available = True
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
            self.reranker_available = False
    
    async def retrieve(self, request: QueryRequest) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for query."""
        logger.info(f"Retrieving for query: {request.query[:100]}...")
        
        # Generate query embedding
        query_embedding = await self._embed_query(request.query)
        
        # Build filter
        filter_dict = request.filters or {}
        if request.modality:
            filter_dict["modality"] = request.modality.value
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=request.top_k * 2 if request.use_reranking else request.top_k,  # Get more for reranking
            namespace=request.namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Extract matches
        chunks = []
        for match in results.matches:
            chunks.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        # Rerank if enabled
        if request.use_reranking and self.reranker_available and len(chunks) > 0:
            chunks = await self._rerank(request.query, chunks, request.top_k)
        else:
            chunks = chunks[:request.top_k]
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    async def _embed_query(self, query: str) -> List[float]:
        """Embed query text."""
        import asyncio
        embedding = await asyncio.to_thread(
            self.embeddings.embed_query,
            query
        )
        return embedding
    
    async def _rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Rerank retrieved chunks using cross-encoder."""
        if not chunks:
            return []
        
        # Prepare pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Score pairs
        import asyncio
        scores = await asyncio.to_thread(
            self.reranker.predict,
            pairs
        )
        
        # Update scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Reranked {len(chunks)} chunks, returning top {top_k}")
        return chunks[:top_k]
    
    def format_context(self, chunks: List[Dict], max_tokens: int = 2000) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        token_count = 0
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            
            # Format citation
            citation = f"[Source {i}"
            if metadata.get("page"):
                citation += f" | Page {metadata['page']}"
            if metadata.get("frame_sec"):
                mins, secs = divmod(int(metadata["frame_sec"]), 60)
                citation += f" | t={mins:02d}:{secs:02d}"
            citation += "]"
            
            # Add chunk
            chunk_text = f"{citation}\n{chunk['text']}\n"
            
            # Rough token count (1 token ≈ 4 chars)
            chunk_tokens = len(chunk_text) // 4
            
            if token_count + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            token_count += chunk_tokens
        
        return "\n".join(context_parts)
