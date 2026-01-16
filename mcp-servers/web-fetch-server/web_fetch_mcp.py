"""MCP Web Fetch Server - Rate-limited web scraping."""
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
from bs4 import BeautifulSoup
from readability import Document
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

app = FastAPI(title="Web Fetch MCP Server")

# Configuration
ALLOWED_DOMAINS = os.getenv("MCP_ALLOWED_DOMAINS", "").split(",")
RATE_LIMIT_REQUESTS = 10  # requests per minute per domain
TIMEOUT_SECONDS = 30

# Rate limiting state
request_counts = defaultdict(list)


class FetchRequest(BaseModel):
    """Request to fetch URL."""
    url: HttpUrl
    extract_main_content: bool = True
    include_images: bool = False


class FetchResponse(BaseModel):
    """Response from fetch."""
    url: str
    title: str
    content: str
    html: Optional[str] = None
    images: List[str] = []
    status_code: int


def is_domain_allowed(url: str) -> bool:
    """Check if domain is in allowlist."""
    if not ALLOWED_DOMAINS or ALLOWED_DOMAINS == ['']:
        return True  # No restrictions if not configured
    
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    
    return any(allowed in domain for allowed in ALLOWED_DOMAINS if allowed)


def check_rate_limit(domain: str) -> bool:
    """Check if domain has exceeded rate limit."""
    now = datetime.now()
    one_minute_ago = now - timedelta(minutes=1)
    
    # Clean old requests
    request_counts[domain] = [
        ts for ts in request_counts[domain]
        if ts > one_minute_ago
    ]
    
    # Check limit
    if len(request_counts[domain]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Record request
    request_counts[domain].append(now)
    return True


async def fetch_url_content(url: str, extract_main: bool = True) -> dict:
    """Fetch and parse URL content."""
    from urllib.parse import urlparse
    
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        
        html = response.text
        
        if extract_main:
            # Extract main content using readability
            doc = Document(html)
            title = doc.title()
            content = doc.summary()
            
            # Convert HTML to text
            soup = BeautifulSoup(content, 'lxml')
            text_content = soup.get_text(separator='\n', strip=True)
        else:
            soup = BeautifulSoup(html, 'lxml')
            title = soup.title.string if soup.title else ""
            text_content = soup.get_text(separator='\n', strip=True)
        
        # Extract images
        images = [
            img['src'] for img in soup.find_all('img')
            if img.get('src')
        ]
        
        return {
            "title": title,
            "content": text_content,
            "html": content if extract_main else html,
            "images": images,
            "status_code": response.status_code
        }


@app.post("/fetch", response_model=FetchResponse)
async def fetch(request: FetchRequest):
    """Fetch URL content."""
    url_str = str(request.url)
    
    # Check domain allowlist
    if not is_domain_allowed(url_str):
        raise HTTPException(
            status_code=403,
            detail=f"Domain not in allowlist. Allowed: {ALLOWED_DOMAINS}"
        )
    
    # Check rate limit
    from urllib.parse import urlparse
    domain = urlparse(url_str).netloc
    
    if not check_rate_limit(domain):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {domain}"
        )
    
    try:
        result = await fetch_url_content(url_str, request.extract_main_content)
        
        if not request.include_images:
            result["images"] = []
        
        return FetchResponse(url=url_str, **result)
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")


@app.post("/fetch_batch")
async def fetch_batch(urls: List[HttpUrl], extract_main_content: bool = True):
    """Fetch multiple URLs."""
    tasks = [
        fetch(FetchRequest(url=url, extract_main_content=extract_main_content))
        for url in urls
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "results": [
            r if not isinstance(r, Exception) else {"error": str(r)}
            for r in results
        ]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "web-fetch-mcp-server",
        "allowed_domains": ALLOWED_DOMAINS if ALLOWED_DOMAINS != [''] else "all"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_WEB_FETCH_SERVER_PORT", 5002))
    uvicorn.run(app, host="0.0.0.0", port=port)
