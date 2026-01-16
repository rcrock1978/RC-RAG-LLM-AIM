"""MCP File Server - Controlled file system access."""
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="File MCP Server")

# Configuration
ALLOWED_DIRS = [
    os.getenv("UPLOAD_DIR", "./uploads"),
    "./data",
    "./docs"
]


class FileInfo(BaseModel):
    """File information."""
    path: str
    name: str
    size: int
    mime_type: Optional[str]
    hash: str


class ListFilesRequest(BaseModel):
    """Request to list files."""
    directory: str
    pattern: Optional[str] = "*"


class ReadFileRequest(BaseModel):
    """Request to read file."""
    path: str


def is_path_allowed(path: str) -> bool:
    """Check if path is within allowed directories."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(os.path.abspath(d)) for d in ALLOWED_DIRS)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@app.post("/list_files", response_model=List[FileInfo])
async def list_files(request: ListFilesRequest):
    """List files in a directory."""
    if not is_path_allowed(request.directory):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(request.directory):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    files = []
    path = Path(request.directory)
    
    for file_path in path.glob(request.pattern):
        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            files.append(FileInfo(
                path=str(file_path),
                name=file_path.name,
                size=file_path.stat().st_size,
                mime_type=mime_type,
                hash=compute_file_hash(str(file_path))
            ))
    
    return files


@app.post("/read_file")
async def read_file(request: ReadFileRequest):
    """Read file content."""
    if not is_path_allowed(request.path):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(request.path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content, "path": request.path}
    except UnicodeDecodeError:
        # Binary file
        with open(request.path, "rb") as f:
            content = f.read()
        return {"content": content.hex(), "path": request.path, "encoding": "hex"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "file-mcp-server"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_FILE_SERVER_PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
