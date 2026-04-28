# Enterprise AI Financial Intelligence Platform

> Multi-agent RAG system on Databricks Lakehouse for autonomous financial document analysis — SEC filings, real-time data verification, and zero hallucination on financial figures.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-red?style=flat-square&logo=databricks)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green?style=flat-square)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange?style=flat-square&logo=amazon-aws)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-teal?style=flat-square&logo=fastapi)

---

## Overview

This platform ingests SEC 10-K/10-Q filings, indexes them into a hybrid vector store, and orchestrates three specialized AI agents (Retriever, Analyst, Synthesizer) to answer complex financial questions with verifiable accuracy — no hallucinated figures.

### Key Results

| Metric | Score |
|---|---|
| RAGAS Faithfulness | **0.87** |
| Answer Relevancy | **0.83** |
| Context Precision | **0.79** |
| Pipeline Coverage | 10,000+ document sections |

---

## Architecture

```
SEC Filings (10-K/10-Q)
        │
        ▼
  Delta Lake Ingestion (Bronze Layer)
        │  semantic chunking
        ▼
  Silver Layer (cleaned, structured chunks)
        │  embedding + indexing
        ▼
  Pinecone (hybrid dense/sparse search)
        │
        ▼
  LangGraph State Machine
  ┌─────┬──────────┬────────────┐
  │     │          │            │
  ▼     ▼          ▼            ▼
Retriever  Analyst  Synthesizer  MCP Server
                                (Gold Layer
                                 verification)
        │
        ▼
  FastAPI Endpoint (Databricks Model Serving)
        │
        ▼
  LangSmith Observability
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data Ingestion** | Delta Lake (open table format), Apache Spark |
| **Medallion Architecture** | Bronze → Silver → Gold on Databricks Lakehouse |
| **Vector Store** | Pinecone (hybrid dense/sparse search) |
| **Agent Orchestration** | LangGraph state machine, 3 specialized agents |
| **Real-time Verification** | MCP Server exposing Gold-layer tables as callable tools |
| **LLM Backend** | Amazon Bedrock |
| **Evaluation** | RAGAS framework, 50 golden Q&A pairs |
| **Serving** | FastAPI via Databricks Model Serving |
| **Observability** | LangSmith (end-to-end agent tracing) |
| **Governance** | Databricks Unity Catalog, MLflow |

---

## Agent Design

### Retriever Agent
- Performs hybrid dense/sparse search against Pinecone
- Pulls semantically similar chunks from 10,000+ indexed SEC sections
- Context precision: **0.79**

### Analyst Agent
- Grounds financial figures against live Gold-layer Delta tables via MCP server
- Eliminates hallucination on revenue, EPS, debt ratios, and other metrics
- Real-time verification — no stale data

### Synthesizer Agent
- Assembles final answer from Retriever context + Analyst-verified figures
- Faithfulness score: **0.87** | Answer relevancy: **0.83**

---

## Evaluation

All evals run automatically in Databricks Workflows on every model or prompt change:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=golden_qa_pairs,          # 50 hand-crafted Q&A pairs
    metrics=[faithfulness, answer_relevancy, context_precision]
)
# faithfulness: 0.87 | answer_relevancy: 0.83 | context_precision: 0.79
```

---

## Project Structure

```
├── ingestion/
│   ├── sec_ingestion.py          # Delta Lake Bronze layer ingestion
│   ├── chunking.py               # Semantic chunking logic
│   └── embeddings.py             # Embedding pipeline
├── agents/
│   ├── graph.py                  # LangGraph state machine
│   ├── retriever.py              # Retriever agent
│   ├── analyst.py                # Analyst agent + MCP client
│   └── synthesizer.py            # Synthesizer agent
├── mcp_server/
│   └── gold_layer_server.py      # MCP server exposing Gold tables
├── evaluation/
│   ├── ragas_eval.py             # RAGAS evaluation harness
│   └── golden_qa_pairs.json      # 50 golden Q&A pairs
├── serving/
│   └── api.py                    # FastAPI endpoint
└── notebooks/
    └── demo.ipynb                # End-to-end demo
```

---

## Getting Started

```bash
# Clone
git clone https://github.com/hemanthkamsani/ai-financial-intelligence-platform
cd ai-financial-intelligence-platform

# Install
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in: DATABRICKS_HOST, DATABRICKS_TOKEN, PINECONE_API_KEY, AWS_REGION

# Run ingestion
python ingestion/sec_ingestion.py --ticker AAPL --years 2022,2023,2024

# Start MCP server
python mcp_server/gold_layer_server.py

# Launch agent
python agents/graph.py --query "What was Apple's revenue growth YoY in FY2024?"

# Run evals
python evaluation/ragas_eval.py
```

---

## Results Demo

```
Query: "What was Microsoft's operating income margin in FY2024 and how did it compare to FY2023?"

Retriever: Found 8 relevant chunks from MSFT 10-K filings (2023, 2024)
Analyst:   Verified via MCP → Gold table: FY2024 OI margin 44.6%, FY2023 41.8%
Synthesizer: Microsoft's operating income margin expanded from 41.8% in FY2023 to
             44.6% in FY2024, a 280 basis point improvement driven by...

Faithfulness: 0.91 | Latency: 4.2s
```

---

## Author

**Hemanth Reddy Kamsani** — Senior Data Engineer  
6+ years | AWS · Databricks · dbt · Kafka · LangGraph · MCP  
📧 kamsanihemanthreddy7@gmail.com  
🔗 [linkedin.com/in/hemanthkamsani](https://linkedin.com/in/hemanthkamsani)

---

*Open to Senior Data Engineer roles — W2 or C2C, anywhere in the USA (remote preferred)*
