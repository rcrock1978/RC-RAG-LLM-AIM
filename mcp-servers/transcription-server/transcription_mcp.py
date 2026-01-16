"""MCP Transcription Server - Audio/Video transcription."""
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import ffmpeg

app = FastAPI(title="Transcription MCP Server")

# Configuration
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "openai")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")


class TranscriptionSegment(BaseModel):
    """Transcription segment."""
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


class TranscriptionRequest(BaseModel):
    """Request for transcription."""
    file_path: str
    language: Optional[str] = None
    return_segments: bool = True


class TranscriptionResponse(BaseModel):
    """Transcription response."""
    text: str
    segments: List[TranscriptionSegment]
    language: Optional[str]
    duration: float


def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video."""
    try:
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',
            ac=1,
            ar='16k'
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        raise Exception(f"Failed to extract audio: {e.stderr.decode()}")


async def transcribe_openai(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe using OpenAI Whisper API."""
    import openai
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            response_format="verbose_json",
            language=language
        )
    
    return {
        "text": transcript.text,
        "segments": [
            {"text": seg.text, "start": seg.start, "end": seg.end}
            for seg in transcript.segments
        ],
        "language": transcript.language,
        "duration": transcript.duration
    }


async def transcribe_local(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe using local Whisper."""
    import whisper
    
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, language=language)
    
    return {
        "text": result["text"],
        "segments": [
            {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
            for seg in result["segments"]
        ],
        "language": result.get("language"),
        "duration": result["segments"][-1]["end"] if result["segments"] else 0
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(request: TranscriptionRequest):
    """Transcribe audio or video file."""
    file_path = request.file_path
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine if video or audio
    ext = Path(file_path).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Extract audio if video
    if is_video:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_path = temp_audio.name
        extract_audio(file_path, audio_path)
    else:
        audio_path = file_path
    
    try:
        # Transcribe
        if WHISPER_BACKEND == "openai":
            result = await transcribe_openai(audio_path, request.language)
        else:
            result = await transcribe_local(audio_path, request.language)
        
        return TranscriptionResponse(**result)
    
    finally:
        # Cleanup temp audio
        if is_video and os.path.exists(audio_path):
            os.unlink(audio_path)


@app.post("/transcribe_upload")
async def transcribe_upload(file: UploadFile = File(...), language: Optional[str] = None):
    """Transcribe uploaded file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        request = TranscriptionRequest(file_path=temp_path, language=language)
        result = await transcribe(request)
        return result
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "transcription-mcp-server",
        "backend": WHISPER_BACKEND,
        "model": WHISPER_MODEL
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_TRANSCRIPTION_SERVER_PORT", 5004))
    uvicorn.run(app, host="0.0.0.0", port=port)
