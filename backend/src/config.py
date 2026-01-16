"""Configuration management."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    OPENAI_EMBEDDING_DIMS = int(os.getenv("OPENAI_EMBEDDING_DIMS", 3072))
    
    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rc-rag-multimodal")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
    PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
    
    # LangSmith
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rc-rag-multimodal")
    
    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
    
    # Retrieval
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 8))
    RETRIEVAL_MMR_DIVERSITY = float(os.getenv("RETRIEVAL_MMR_DIVERSITY", 0.3))
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # File uploads
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 524288000))
    
    # Video processing
    FFMPEG_FRAME_INTERVAL = int(os.getenv("FFMPEG_FRAME_INTERVAL", 2))
    FFMPEG_MAX_FRAMES = int(os.getenv("FFMPEG_MAX_FRAMES", 100))
    CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-L/14")
    
    # Whisper
    WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "openai")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
    
    # MCP Servers
    MCP_FILE_SERVER_PORT = int(os.getenv("MCP_FILE_SERVER_PORT", 5001))
    MCP_WEB_FETCH_SERVER_PORT = int(os.getenv("MCP_WEB_FETCH_SERVER_PORT", 5002))
    MCP_PINECONE_SERVER_PORT = int(os.getenv("MCP_PINECONE_SERVER_PORT", 5003))
    MCP_TRANSCRIPTION_SERVER_PORT = int(os.getenv("MCP_TRANSCRIPTION_SERVER_PORT", 5004))
    MCP_VISION_SERVER_PORT = int(os.getenv("MCP_VISION_SERVER_PORT", 5005))
    MCP_ALLOWED_DOMAINS = os.getenv("MCP_ALLOWED_DOMAINS", "").split(",")
    
    # Server
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
    BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
