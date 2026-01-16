"""Data schemas for RAG system."""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Source type enumeration."""
    URL = "url"
    PDF = "pdf"
    VIDEO = "video"
    TEXT = "text"


class Modality(str, Enum):
    """Content modality enumeration."""
    TEXT = "text"
    FRAME = "frame"
    AUDIO = "audio"


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    source: SourceType
    uri: str
    modality: Modality
    page: Optional[int] = None
    frame_sec: Optional[float] = None
    chunk_id: str
    hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mime: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Document chunk with content and metadata."""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class IngestRequest(BaseModel):
    """Request to ingest a document."""
    uri: str
    source_type: SourceType
    namespace: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from ingestion."""
    success: bool
    chunk_count: int
    index_name: str
    namespace: str
    message: str


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str
    top_k: int = 8
    namespace: str = "default"
    filters: Optional[Dict[str, Any]] = None
    modality: Optional[Modality] = None
    use_reranking: bool = True


class Citation(BaseModel):
    """Citation for a retrieved chunk."""
    source: str
    uri: str
    page: Optional[int] = None
    frame_sec: Optional[float] = None
    relevance_score: float
    content_preview: str


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    citations: List[Citation]
    context_used: int
    tokens_used: Optional[int] = None
    latency_ms: float


class TranscriptionSegment(BaseModel):
    """Segment from audio/video transcription."""
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


class VideoFrame(BaseModel):
    """Extracted video frame."""
    frame_sec: float
    image_path: str
    caption: Optional[str] = None
    embedding: Optional[List[float]] = None


class HealthStatus(BaseModel):
    """Health status of the system."""
    status: str
    pinecone_connected: bool
    llm_available: bool
    embedding_available: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
