"""MCP Vision Server - Frame extraction and CLIP embeddings."""
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ffmpeg
from PIL import Image
import torch

app = FastAPI(title="Vision MCP Server")

# Configuration
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-L/14")
FRAME_INTERVAL = int(os.getenv("FFMPEG_FRAME_INTERVAL", 2))
MAX_FRAMES = int(os.getenv("FFMPEG_MAX_FRAMES", 100))

# Load CLIP model
try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained='openai'
    )
    clip_model.eval()
    CLIP_AVAILABLE = True
except Exception as e:
    print(f"Warning: CLIP not available: {e}")
    CLIP_AVAILABLE = False


class FrameExtractionRequest(BaseModel):
    """Request to extract frames."""
    video_path: str
    interval: Optional[float] = None
    max_frames: Optional[int] = None
    strategy: str = "interval"  # interval or scene-detect


class VideoFrame(BaseModel):
    """Extracted frame."""
    frame_sec: float
    frame_path: str


class FrameEmbeddingRequest(BaseModel):
    """Request to embed frames."""
    frame_paths: List[str]
    generate_captions: bool = False


class FrameEmbedding(BaseModel):
    """Frame with embedding."""
    frame_path: str
    embedding: List[float]
    caption: Optional[str] = None


def extract_frames_interval(video_path: str, interval: float, max_frames: int, output_dir: str) -> List[VideoFrame]:
    """Extract frames at regular intervals."""
    frames = []
    
    # Get video duration
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])
    
    # Calculate frame times
    frame_times = []
    t = 0
    while t < duration and len(frame_times) < max_frames:
        frame_times.append(t)
        t += interval
    
    # Extract frames
    for i, t in enumerate(frame_times):
        output_path = os.path.join(output_dir, f"frame_{i:04d}_{t:.2f}s.jpg")
        
        try:
            ffmpeg.input(video_path, ss=t).output(
                output_path,
                vframes=1,
                format='image2',
                vcodec='mjpeg'
            ).overwrite_output().run(quiet=True)
            
            frames.append(VideoFrame(frame_sec=t, frame_path=output_path))
        except ffmpeg.Error as e:
            print(f"Error extracting frame at {t}s: {e.stderr.decode()}")
    
    return frames


def extract_frames_scene_detect(video_path: str, max_frames: int, output_dir: str) -> List[VideoFrame]:
    """Extract frames using scene detection."""
    # Simple scene detection using ffmpeg select filter
    frames = []
    
    try:
        # Use ffmpeg scene detection
        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
        
        ffmpeg.input(video_path).output(
            output_pattern,
            vf=f'select=gt(scene\\,0.3)',
            vsync='vfr',
            vframes=max_frames
        ).overwrite_output().run(quiet=True)
        
        # Get frame timestamps
        for frame_file in sorted(Path(output_dir).glob("frame_*.jpg")):
            # Extract timestamp from metadata if available
            frame_sec = float(frame_file.stem.split('_')[1]) if '_' in frame_file.stem else 0
            frames.append(VideoFrame(frame_sec=frame_sec, frame_path=str(frame_file)))
    
    except ffmpeg.Error as e:
        print(f"Scene detection failed: {e.stderr.decode()}, falling back to interval")
        return extract_frames_interval(video_path, FRAME_INTERVAL, max_frames, output_dir)
    
    return frames


def embed_image_clip(image_path: str) -> List[float]:
    """Generate CLIP embedding for image."""
    if not CLIP_AVAILABLE:
        raise Exception("CLIP model not available")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = clip_preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze().cpu().numpy().tolist()


@app.post("/extract_frames", response_model=List[VideoFrame])
async def extract_frames(request: FrameExtractionRequest):
    """Extract frames from video."""
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    interval = request.interval or FRAME_INTERVAL
    max_frames = request.max_frames or MAX_FRAMES
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="frames_")
    
    try:
        if request.strategy == "scene-detect":
            frames = extract_frames_scene_detect(request.video_path, max_frames, temp_dir)
        else:
            frames = extract_frames_interval(request.video_path, interval, max_frames, temp_dir)
        
        return frames
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {str(e)}")


@app.post("/embed_frames", response_model=List[FrameEmbedding])
async def embed_frames(request: FrameEmbeddingRequest):
    """Generate embeddings for frames."""
    if not CLIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="CLIP model not available")
    
    results = []
    
    for frame_path in request.frame_paths:
        if not os.path.exists(frame_path):
            continue
        
        try:
            embedding = embed_image_clip(frame_path)
            
            # Optional: Generate caption using CLIP
            caption = None
            if request.generate_captions:
                # Simple caption generation (could be enhanced with BLIP/etc)
                caption = f"Frame from video"
            
            results.append(FrameEmbedding(
                frame_path=frame_path,
                embedding=embedding,
                caption=caption
            ))
        
        except Exception as e:
            print(f"Error embedding frame {frame_path}: {e}")
    
    return results


@app.post("/extract_and_embed")
async def extract_and_embed(request: FrameExtractionRequest):
    """Extract frames and generate embeddings in one call."""
    # Extract frames
    frames = await extract_frames(request)
    
    # Embed frames
    frame_paths = [f.frame_path for f in frames]
    embeddings = await embed_frames(FrameEmbeddingRequest(frame_paths=frame_paths))
    
    return {
        "frames": frames,
        "embeddings": embeddings
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "vision-mcp-server",
        "clip_available": CLIP_AVAILABLE,
        "model": CLIP_MODEL_NAME
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_VISION_SERVER_PORT", 5005))
    uvicorn.run(app, host="0.0.0.0", port=port)
