"""Chat Agent - Handles LLM generation with citations."""
from typing import List, Dict, Any, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from ..config import settings
from ..schemas import QueryRequest, QueryResponse, Citation
from ..logging_config import get_logger
from .retrieval_agent import RetrievalAgent

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a grounded AI assistant. Your responses must be based solely on the provided context.

GUIDELINES:
1. Answer questions using ONLY the information in the context below
2. Cite sources using the format [Source N] where N is the source number
3. If the context doesn't contain sufficient information, clearly state: "I don't have enough information to answer this question based on the available sources."
4. For video sources with timestamps, reference them as [Source N | t=MM:SS]
5. Be concise and precise
6. Do not make assumptions or add information not in the context

CONTEXT:
{context}
"""


class ChatAgent:
    """Agent for chat and answer generation."""
    
    def __init__(self):
        """Initialize chat agent."""
        self.retrieval_agent = RetrievalAgent()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True
        )
        
        # Backup LLM (Anthropic)
        try:
            self.backup_llm = ChatAnthropic(
                model=settings.ANTHROPIC_MODEL,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.1
            )
        except:
            self.backup_llm = None
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
    
    async def chat(self, request: QueryRequest) -> QueryResponse:
        """Generate answer for query."""
        import time
        start_time = time.time()
        
        # Retrieve relevant chunks
        chunks = await self.retrieval_agent.retrieve(request)
        
        if not chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                context_used=0,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Format context
        context = self.retrieval_agent.format_context(chunks, max_tokens=2000)
        
        # Generate answer
        messages = self.prompt_template.format_messages(
            context=context,
            question=request.query
        )
        
        try:
            response = await self.llm.ainvoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            if self.backup_llm:
                logger.info("Falling back to Anthropic")
                response = await self.backup_llm.ainvoke(messages)
                answer = response.content
            else:
                raise
        
        # Extract citations
        citations = self._build_citations(chunks)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            context_used=len(chunks),
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )
    
    async def chat_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """Stream answer generation."""
        # Retrieve chunks
        chunks = await self.retrieval_agent.retrieve(request)
        
        if not chunks:
            yield "I couldn't find any relevant information to answer your question."
            return
        
        # Format context
        context = self.retrieval_agent.format_context(chunks, max_tokens=2000)
        
        # Generate streaming response
        messages = self.prompt_template.format_messages(
            context=context,
            question=request.query
        )
        
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    def _build_citations(self, chunks: List[Dict]) -> List[Citation]:
        """Build citation objects from chunks."""
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            
            # Get content preview
            preview = chunk["text"][:200]
            if len(chunk["text"]) > 200:
                preview += "..."
            
            citation = Citation(
                source=f"Source {i}",
                uri=metadata.get("uri", ""),
                page=metadata.get("page"),
                frame_sec=metadata.get("frame_sec"),
                relevance_score=chunk.get("rerank_score", chunk.get("score", 0.0)),
                content_preview=preview
            )
            
            citations.append(citation)
        
        return citations
