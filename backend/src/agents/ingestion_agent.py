"""Ingestion Agent - Orchestrates document ingestion pipeline."""
import os
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ..config import settings
from ..schemas import (
    IngestRequest, IngestResponse, DocumentChunk, ChunkMetadata,
    SourceType, Modality
)
from ..logging_config import get_logger

logger = get_logger(__name__)


class IngestionAgent:
    """Agent for document ingestion."""
    
    def __init__(self):
        """Initialize ingestion agent."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        
        # MCP service URLs
        self.file_server = f"http://localhost:{settings.MCP_FILE_SERVER_PORT}"
        self.web_server = f"http://localhost:{settings.MCP_WEB_FETCH_SERVER_PORT}"
        self.transcription_server = f"http://localhost:{settings.MCP_TRANSCRIPTION_SERVER_PORT}"
        self.vision_server = f"http://localhost:{settings.MCP_VISION_SERVER_PORT}"
    
    async def ingest_url(self, uri: str, namespace: str = "default", metadata: Dict[str, Any] = None) -> IngestResponse:
        """Ingest web URL."""
        logger.info(f"Ingesting URL: {uri}")
        
        async with httpx.AsyncClient() as client:
            # Fetch via MCP web server
            response = await client.post(
                f"{self.web_server}/fetch",
                json={"url": uri, "extract_main_content": True}
            )
            response.raise_for_status()
            data = response.json()
        
        # Create chunks
        text_content = data["content"]
        text_chunks = self.text_splitter.split_text(text_content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_meta = ChunkMetadata(
                source=SourceType.URL,
                uri=uri,
                modality=Modality.TEXT,
                chunk_id=f"{hash(uri)}_{i}",
                hash=hash(chunk_text).__str__(),
                title=data.get("title"),
                mime="text/html"
            )
            
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_meta
            ))
        
        # Embed and upsert
        await self._embed_and_store(chunks, namespace)
        
        return IngestResponse(
            success=True,
            chunk_count=len(chunks),
            index_name=settings.PINECONE_INDEX_NAME,
            namespace=namespace,
            message=f"Successfully ingested {len(chunks)} chunks from URL"
        )
    
    async def ingest_pdf(self, file_path: str, namespace: str = "default", metadata: Dict[str, Any] = None) -> IngestResponse:
        """Ingest PDF file."""
        logger.info(f"Ingesting PDF: {file_path}")
        
        from pypdf import PdfReader
        
        reader = PdfReader(file_path)
        chunks = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            page_chunks = self.text_splitter.split_text(text)
            
            for i, chunk_text in enumerate(page_chunks):
                chunk_meta = ChunkMetadata(
                    source=SourceType.PDF,
                    uri=file_path,
                    modality=Modality.TEXT,
                    page=page_num + 1,
                    chunk_id=f"{hash(file_path)}_{page_num}_{i}",
                    hash=hash(chunk_text).__str__(),
                    title=Path(file_path).stem,
                    mime="application/pdf"
                )
                
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_meta
                ))
        
        await self._embed_and_store(chunks, namespace)
        
        return IngestResponse(
            success=True,
            chunk_count=len(chunks),
            index_name=settings.PINECONE_INDEX_NAME,
            namespace=namespace,
            message=f"Successfully ingested {len(chunks)} chunks from PDF"
        )
    
    async def ingest_video(self, file_path: str, namespace: str = "default", metadata: Dict[str, Any] = None) -> IngestResponse:
        """Ingest video file with transcription and frame embeddings."""
        logger.info(f"Ingesting video: {file_path}")
        
        chunks = []
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            # 1. Transcribe audio
            logger.info("Transcribing video...")
            transcript_response = await client.post(
                f"{self.transcription_server}/transcribe",
                json={"file_path": file_path, "return_segments": True}
            )
            transcript_response.raise_for_status()
            transcript_data = transcript_response.json()
            
            # Create transcript chunks
            for segment in transcript_data["segments"]:
                chunk_meta = ChunkMetadata(
                    source=SourceType.VIDEO,
                    uri=file_path,
                    modality=Modality.TEXT,
                    frame_sec=segment["start"],
                    chunk_id=f"{hash(file_path)}_transcript_{segment['start']}",
                    hash=hash(segment['text']).__str__(),
                    title=Path(file_path).stem,
                    mime="video/mp4",
                    extra={"end_sec": segment["end"]}
                )
                
                chunks.append(DocumentChunk(
                    content=segment["text"],
                    metadata=chunk_meta
                ))
            
            # 2. Extract and embed frames
            logger.info("Extracting and embedding frames...")
            frames_response = await client.post(
                f"{self.vision_server}/extract_and_embed",
                json={
                    "video_path": file_path,
                    "strategy": "interval"
                }
            )
            frames_response.raise_for_status()
            frames_data = frames_response.json()
            
            # Store frame embeddings separately
            frame_vectors = []
            for frame_data, embed_data in zip(
                frames_data["frames"],
                frames_data["embeddings"]
            ):
                frame_meta = ChunkMetadata(
                    source=SourceType.VIDEO,
                    uri=file_path,
                    modality=Modality.FRAME,
                    frame_sec=frame_data["frame_sec"],
                    chunk_id=f"{hash(file_path)}_frame_{frame_data['frame_sec']}",
                    hash=hash(frame_data["frame_path"]).__str__(),
                    title=Path(file_path).stem,
                    mime="image/jpeg"
                )
                
                frame_vectors.append({
                    "id": frame_meta.chunk_id,
                    "values": embed_data["embedding"],
                    "metadata": frame_meta.model_dump()
                })
            
            # Upsert frame vectors
            if frame_vectors:
                self.index.upsert(vectors=frame_vectors, namespace=namespace)
        
        # Embed and store transcript chunks
        await self._embed_and_store(chunks, namespace)
        
        total_chunks = len(chunks) + len(frame_vectors)
        
        return IngestResponse(
            success=True,
            chunk_count=total_chunks,
            index_name=settings.PINECONE_INDEX_NAME,
            namespace=namespace,
            message=f"Successfully ingested video: {len(chunks)} transcript segments, {len(frame_vectors)} frames"
        )
    
    async def _embed_and_store(self, chunks: List[DocumentChunk], namespace: str):
        """Embed chunks and store in Pinecone."""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await asyncio.to_thread(
            self.embeddings.embed_documents,
            texts
        )
        
        # Prepare vectors for Pinecone
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.metadata.chunk_id,
                "values": embedding,
                "metadata": {
                    **chunk.metadata.model_dump(),
                    "text": chunk.content  # Store text in metadata
                }
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
        
        logger.info(f"Stored {len(vectors)} vectors in namespace '{namespace}'")
    
    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Ingest document based on source type."""
        try:
            if request.source_type == SourceType.URL:
                return await self.ingest_url(request.uri, request.namespace, request.metadata)
            
            elif request.source_type == SourceType.PDF:
                return await self.ingest_pdf(request.uri, request.namespace, request.metadata)
            
            elif request.source_type == SourceType.VIDEO:
                return await self.ingest_video(request.uri, request.namespace, request.metadata)
            
            else:
                raise ValueError(f"Unsupported source type: {request.source_type}")
        
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return IngestResponse(
                success=False,
                chunk_count=0,
                index_name=settings.PINECONE_INDEX_NAME,
                namespace=request.namespace,
                message=f"Ingestion failed: {str(e)}"
            )
