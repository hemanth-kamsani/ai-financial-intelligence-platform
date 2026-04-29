"""
FastAPI Serving Endpoint
Deployed via Databricks Model Serving with LangSmith observability.
"""

import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langsmith import traceable, Client as LangSmithClient
from agents.graph import run_query
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "financial-intelligence-platform"

app = FastAPI(
    title="Enterprise AI Financial Intelligence Platform",
    description="Multi-agent RAG system for SEC filing analysis with zero hallucination on financial figures",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str
    max_sources: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    confidence: float
    latency_ms: float


@traceable(name="financial-query-endpoint")
@app.post("/query", response_model=QueryResponse)
async def query_financials(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    logger.info(f"[API] Query: {request.query[:80]}...")
    start = time.time()

    try:
        result = run_query(request.query)
    except Exception as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Pipeline execution failed")

    latency_ms = round((time.time() - start) * 1000, 1)

    return QueryResponse(
        query=request.query,
        answer=result["final_answer"],
        sources=result["sources"][:request.max_sources],
        confidence=result["confidence"],
        latency_ms=latency_ms
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
async def root():
    return {"message": "Enterprise AI Financial Intelligence Platform", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
