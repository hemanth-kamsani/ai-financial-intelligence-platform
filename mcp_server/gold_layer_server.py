"""
MCP Server — Gold Layer Financial Metrics
Exposes Databricks Gold-layer Delta tables as callable tools for AI agents.
Governed via Databricks Unity Catalog + AWS Lake Formation.

This is the key component that eliminates hallucination on financial figures:
agents call this server to verify any number before including it in an answer.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from databricks import sql
from databricks.sdk import WorkspaceClient
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Intelligence MCP Server",
    description="MCP server exposing Gold-layer Delta tables as callable tools for AI agents",
    version="1.0.0"
)

DATABRICKS_HOST  = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
HTTP_PATH        = os.environ["DATABRICKS_HTTP_PATH"]

CATALOG   = "prod"
SCHEMA    = "gold"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.financial_metrics"
FILINGS_TABLE = f"{CATALOG}.{SCHEMA}.sec_filings_summary"


def get_connection():
    return sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )


class ToolCallRequest(BaseModel):
    tool: str
    parameters: dict


def verify_financial_metric(ticker: str, metric: str, fiscal_year: str) -> dict:
    query = f"""
        SELECT ticker, metric_name, metric_value, unit, fiscal_year, filing_type, updated_at
        FROM {METRICS_TABLE}
        WHERE ticker = '{ticker.upper()}'
          AND metric_name = '{metric.lower()}'
          AND fiscal_year = '{fiscal_year}'
        ORDER BY updated_at DESC
        LIMIT 1
    """
    logger.info(f"[MCP] Querying Gold table: {ticker} | {metric} | {fiscal_year}")
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
        if row:
            return {
                "verified": True,
                "value": str(row[2]),
                "unit": row[3],
                "ticker": row[0],
                "metric": row[1],
                "fiscal_year": row[4],
                "source_table": METRICS_TABLE,
                "as_of_date": str(row[6]) if row[6] else None
            }
        else:
            return {
                "verified": False,
                "value": None,
                "unit": "",
                "ticker": ticker,
                "metric": metric,
                "fiscal_year": fiscal_year,
                "source_table": METRICS_TABLE,
                "as_of_date": None
            }
    except Exception as e:
        logger.error(f"[MCP] Databricks query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def list_available_metrics(ticker: str) -> dict:
    query = f"""
        SELECT DISTINCT metric_name, fiscal_year
        FROM {METRICS_TABLE}
        WHERE ticker = '{ticker.upper()}'
        ORDER BY fiscal_year DESC, metric_name
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    return {
        "ticker": ticker,
        "available_metrics": [{"metric": r[0], "fiscal_year": r[1]} for r in rows]
    }


TOOL_REGISTRY = {
    "verify_financial_metric": verify_financial_metric,
    "list_available_metrics": list_available_metrics,
}


@app.post("/tools/call", response_model=dict)
async def call_tool(request: ToolCallRequest):
    logger.info(f"[MCP] Tool call: {request.tool} | params: {request.parameters}")
    if request.tool not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool}' not found.")
    handler = TOOL_REGISTRY[request.tool]
    return handler(**request.parameters)


@app.get("/tools/list")
async def list_tools():
    return {
        "tools": [
            {
                "name": "verify_financial_metric",
                "description": "Verify a financial figure against the Gold-layer Delta table",
                "parameters": {
                    "ticker": "string — e.g. AAPL, MSFT",
                    "metric": "string — e.g. revenue, margin, eps",
                    "fiscal_year": "string — e.g. FY2024"
                }
            },
            {
                "name": "list_available_metrics",
                "description": "List all metrics available for a given ticker",
                "parameters": {"ticker": "string"}
            }
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "catalog": CATALOG, "schema": SCHEMA}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
