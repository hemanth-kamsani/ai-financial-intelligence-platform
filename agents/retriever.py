"""
Retriever Agent
Performs hybrid dense/sparse search against Pinecone to pull
relevant SEC filing chunks for a given financial query.
"""

import os
from typing import List
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import boto3
import json
import logging

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
bm25 = BM25Encoder().default()

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
TOP_K = 8
ALPHA = 0.7   # weight for dense vs sparse (1.0 = pure dense, 0.0 = pure sparse)


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_dense_embedding(text: str) -> List[float]:
    """Generate dense embedding via Amazon Bedrock Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def get_sparse_embedding(text: str) -> dict:
    """Generate sparse BM25 embedding."""
    sparse = bm25.encode_queries(text)
    return {"indices": sparse["indices"], "values": sparse["values"]}


# ── Search ────────────────────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Perform hybrid dense + sparse search against Pinecone.
    Returns list of matching chunks with metadata.
    """
    dense_vec = get_dense_embedding(query)
    sparse_vec = get_sparse_embedding(query)

    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        alpha=ALPHA,
        include_metadata=True
    )

    chunks = []
    for match in results.get("matches", []):
        chunks.append({
            "chunk_id": match["id"],
            "score": match["score"],
            "text": match["metadata"].get("text", ""),
            "source": match["metadata"].get("source", ""),
            "ticker": match["metadata"].get("ticker", ""),
            "filing_type": match["metadata"].get("filing_type", ""),
            "fiscal_year": match["metadata"].get("fiscal_year", ""),
            "section": match["metadata"].get("section", "")
        })

    logger.info(f"[Retriever] Top match score: {chunks[0]['score']:.3f}" if chunks else "[Retriever] No matches")
    return chunks


def retriever_agent(query: str) -> List[dict]:
    """
    Main entry point for the Retriever agent.
    Called by the LangGraph state machine.
    """
    logger.info(f"[Retriever] Running hybrid search for: {query[:60]}...")
    chunks = hybrid_search(query)

    # Filter low-confidence results
    filtered = [c for c in chunks if c["score"] > 0.65]
    logger.info(f"[Retriever] {len(filtered)} chunks passed score threshold (>0.65)")
    return filtered


if __name__ == "__main__":
    # Quick test
    test_query = "Apple revenue FY2024 year over year growth"
    results = retriever_agent(test_query)
    for r in results[:3]:
        print(f"[{r['score']:.3f}] {r['ticker']} {r['filing_type']} {r['fiscal_year']}")
        print(f"  {r['text'][:120]}...\n")
