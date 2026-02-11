from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import redis
import json
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title="PodPress AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

rag_pipeline = None
llm_inference = None
tts_engine = None


class NewsletterRequest(BaseModel):
    newsletter_url: Optional[str] = None
    newsletter_text: Optional[str] = None
    style: str = "conversational"
    target_length: int = 5000


class PodcastResponse(BaseModel):
    podcast_id: str
    script: str
    audio_url: Optional[str] = None
    processing_time_ms: float
    word_count: int


@app.on_event("startup")
async def startup_event():
    global rag_pipeline, llm_inference, tts_engine
    logger.info("Initializing PodPress AI...")
    # Initialize components
    logger.info("PodPress AI ready")


@app.post("/generate-podcast", response_model=PodcastResponse)
async def generate_podcast(request: NewsletterRequest, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    if not request.newsletter_text and not request.newsletter_url:
        raise HTTPException(status_code=400, detail="Either newsletter_text or newsletter_url required")
    
    podcast_id = f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        if request.newsletter_url:
            import requests
            response = requests.get(request.newsletter_url)
            newsletter_text = response.text
        else:
            newsletter_text = request.newsletter_text
        
        if rag_pipeline:
            rag_result = rag_pipeline.process_newsletter(newsletter_text)
            context = rag_result['retrieved_context']
        else:
            context = newsletter_text
        
        if llm_inference:
            script = await llm_inference.generate_batch_async([context])
            script = script[0] if script else ""
        else:
            script = context[:request.target_length]
        
        word_count = len(script.split())
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            'podcast_id': podcast_id,
            'script': script,
            'processing_time_ms': processing_time,
            'word_count': word_count
        }
        
        if tts_engine:
            background_tasks.add_task(generate_audio, podcast_id, script)
        
        redis_client.setex(f"podcast:{podcast_id}", 86400, json.dumps(result))
        
        return PodcastResponse(**result)
    
    except Exception as e:
        logger.error(f"Podcast generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/podcast/{podcast_id}")
async def get_podcast(podcast_id: str):
    cached = redis_client.get(f"podcast:{podcast_id}")
    if cached:
        return json.loads(cached)
    raise HTTPException(status_code=404, detail="Podcast not found")


async def generate_audio(podcast_id: str, script: str):
    if tts_engine:
        audio_url = tts_engine.generate(script, podcast_id)
        redis_client.set(f"podcast:{podcast_id}:audio", audio_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
