"""
SEC Filing Ingestion — Bronze Layer
Downloads SEC 10-K and 10-Q filings and ingests them into
the Delta Lake Bronze layer using Apache Spark on Databricks.
"""

import os
import re
import time
import requests
import logging
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from delta.tables import DeltaTable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_HEADERS    = {"User-Agent": os.environ.get("SEC_USER_AGENT", "hemanth@email.com")}
BRONZE_PATH    = os.environ.get("BRONZE_PATH", "s3://your-bucket/bronze/sec_filings/")
CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
REQUEST_DELAY  = 0.15

spark = SparkSession.builder.appName("sec-ingestion-bronze").getOrCreate()


def get_company_cik(ticker: str) -> str:
    url = f"{SEC_EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    return resp.json()["cik"]


def get_filing_urls(cik: str, filing_type: str = "10-K", count: int = 3) -> List[dict]:
    url = f"{SEC_EDGAR_BASE}/submissions/CIK{str(cik).zfill(10)}.json"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    data = resp.json()
    filings    = data.get("filings", {}).get("recent", {})
    forms      = filings.get("form", [])
    dates      = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    results = []
    for form, date, acc in zip(forms, dates, accessions):
        if form == filing_type and len(results) < count:
            acc_clean = acc.replace("-", "")
            results.append({
                "filing_type": form,
                "filing_date": date,
                "accession": acc,
                "url": f"{SEC_EDGAR_BASE}/Archives/edgar/data/{cik}/{acc_clean}/{acc}-index.htm"
            })
    logger.info(f"Found {len(results)} {filing_type} filings for CIK {cik}")
    return results


def download_filing_text(url: str) -> str:
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", resp.text)
    return re.sub(r"\s+", " ", text).strip()


def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        token_approx = int(len(sentence.split()) / 0.75)
        if current_len + token_approx > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_tokens, overlap_sentences = 0, []
            for s in reversed(current):
                overlap_tokens += int(len(s.split()) / 0.75)
                overlap_sentences.insert(0, s)
                if overlap_tokens >= overlap:
                    break
            current = overlap_sentences
            current_len = sum(int(len(s.split()) / 0.75) for s in current)
        current.append(sentence)
        current_len += token_approx
    if current:
        chunks.append(" ".join(current))
    return chunks


def write_to_bronze(chunks: List[dict]):
    df = spark.createDataFrame(chunks).withColumn("ingested_at", current_timestamp())
    if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
        dt = DeltaTable.forPath(spark, BRONZE_PATH)
        dt.alias("target").merge(
            df.alias("source"), "target.chunk_id = source.chunk_id"
        ).whenNotMatchedInsertAll().execute()
    else:
        df.write.format("delta").mode("overwrite").save(BRONZE_PATH)
    logger.info(f"[Bronze] Wrote {len(chunks)} chunks")


def ingest_ticker(ticker: str, filing_type: str = "10-K", years: int = 3):
    logger.info(f"[Ingestion] Starting {filing_type} ingestion for {ticker}")
    cik = get_company_cik(ticker)
    filings = get_filing_urls(cik, filing_type, count=years)
    all_chunks = []
    for filing in filings:
        text = download_filing_text(filing["url"])
        chunks = semantic_chunk(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{ticker}_{filing['accession']}_{i:04d}",
                "ticker": ticker,
                "filing_type": filing["filing_type"],
                "filing_date": filing["filing_date"],
                "fiscal_year": f"FY{filing['filing_date'][:4]}",
                "accession": filing["accession"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk_text,
                "source": filing["url"],
                "layer": "bronze"
            })
        logger.info(f"[Ingestion] {ticker} {filing['filing_date']}: {len(chunks)} chunks")
    write_to_bronze(all_chunks)
    return len(all_chunks)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--type", default="10-K")
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()
    total = ingest_ticker(args.ticker, args.type, args.years)
    print(f"Ingested {total} chunks for {args.ticker} into Bronze layer.")
