"""
Analyst Agent
Verifies financial figures by calling the MCP server, which exposes
Databricks Gold-layer Delta tables as callable tools.
Zero hallucination on revenue, margins, EPS, and other financial metrics.
"""

import os
import re
import json
import httpx
import logging
from typing import List

logger = logging.getLogger(__name__)

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001")
MCP_TIMEOUT = 10  # seconds


# ── Figure extraction ─────────────────────────────────────────────────────────

FINANCIAL_PATTERNS = {
    "revenue":        r"\$[\d,.]+\s*(?:billion|million|B|M)",
    "margin":         r"\d+\.?\d*\s*%",
    "eps":            r"EPS[:\s]+\$[\d.]+",
    "operating_income": r"operating income[:\s]+\$[\d,.]+\s*(?:billion|million|B|M)",
    "net_income":     r"net income[:\s]+\$[\d,.]+\s*(?:billion|million|B|M)",
    "yoy_growth":     r"[\d.]+%\s*(?:year-over-year|YoY|y\/y)",
}


def extract_financial_mentions(query: str, chunks: List[dict]) -> List[dict]:
    """
    Extract ticker symbols and financial metric mentions from query + chunks
    to know what to verify via MCP.
    """
    # Common ticker symbols in SEC filings
    ticker_pattern = r'\b(AAPL|MSFT|GOOGL|AMZN|META|NVDA|JPM|BAC|GS|WFC|JNJ|PFE)\b'
    tickers = re.findall(ticker_pattern, " ".join([query] + [c["text"] for c in chunks[:3]]).upper())
    tickers = list(set(tickers))

    mentions = []
    for chunk in chunks[:5]:
        for metric, pattern in FINANCIAL_PATTERNS.items():
            matches = re.findall(pattern, chunk["text"], re.IGNORECASE)
            if matches:
                mentions.append({
                    "ticker": chunk.get("ticker", tickers[0] if tickers else "UNKNOWN"),
                    "metric": metric,
                    "raw_value": matches[0],
                    "fiscal_year": chunk.get("fiscal_year", ""),
                    "source_chunk": chunk["chunk_id"]
                })

    logger.info(f"[Analyst] Extracted {len(mentions)} financial mentions to verify")
    return mentions


# ── MCP server call ───────────────────────────────────────────────────────────

def call_mcp_server(ticker: str, metric: str, fiscal_year: str) -> dict:
    """
    Call MCP server to verify a financial figure against the Gold-layer Delta table.
    MCP server exposes Gold tables as callable tools — returns authoritative values.
    """
    payload = {
        "tool": "verify_financial_metric",
        "parameters": {
            "ticker": ticker,
            "metric": metric,
            "fiscal_year": fiscal_year
        }
    }

    try:
        response = httpx.post(
            f"{MCP_SERVER_URL}/tools/call",
            json=payload,
            timeout=MCP_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"[Analyst] MCP verified: {ticker} {metric} {fiscal_year} → {result.get('value')}")
        return result

    except httpx.TimeoutException:
        logger.warning(f"[Analyst] MCP timeout for {ticker} {metric} — using retrieved value")
        return {"verified": False, "value": None, "error": "timeout"}

    except httpx.HTTPStatusError as e:
        logger.error(f"[Analyst] MCP error {e.response.status_code}: {e}")
        return {"verified": False, "value": None, "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def analyst_agent(query: str, chunks: List[dict]) -> dict:
    """
    Main entry point for the Analyst agent.
    Returns dict of verified financial figures keyed by metric name.
    Called by the LangGraph state machine.
    """
    mentions = extract_financial_mentions(query, chunks)

    verified_figures = {}

    for mention in mentions:
        key = f"{mention['ticker']}_{mention['metric']}_{mention['fiscal_year']}"

        if key in verified_figures:
            continue  # already verified this one

        mcp_result = call_mcp_server(
            ticker=mention["ticker"],
            metric=mention["metric"],
            fiscal_year=mention["fiscal_year"]
        )

        verified_figures[key] = {
            "ticker": mention["ticker"],
            "metric": mention["metric"],
            "fiscal_year": mention["fiscal_year"],
            "retrieved_value": mention["raw_value"],
            "verified_value": mcp_result.get("value"),
            "verified": mcp_result.get("verified", False),
            "source_table": mcp_result.get("source_table", "gold.financial_metrics"),
            "unit": mcp_result.get("unit", ""),
        }

    logger.info(f"[Analyst] Verification complete: {sum(1 for v in verified_figures.values() if v['verified'])} verified, "
                f"{sum(1 for v in verified_figures.values() if not v['verified'])} unverified")

    return verified_figures


if __name__ == "__main__":
    # Quick test (requires MCP server running)
    test_chunks = [
        {
            "chunk_id": "chunk_001",
            "ticker": "AAPL",
            "fiscal_year": "FY2024",
            "text": "Apple reported revenue of $391.0 billion in FY2024, representing 2% YoY growth.",
            "filing_type": "10-K",
            "score": 0.91
        }
    ]
    result = analyst_agent("What was Apple's revenue in FY2024?", test_chunks)
    print(json.dumps(result, indent=2))
