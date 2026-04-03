# Project: Autonomous Regulatory Auditor (ARA) v2.0

**Tagline:** A Multi-Agent LangGraph System for Automated Legal Compliance in SE Asia.

## 1. Executive Summary

ARA 2.0 is an enterprise-grade compliance engine designed to audit corporate documents against dynamic Southeast Asian regulations (focused on Myanmar CBM & MOTC). Unlike standard RAG apps, ARA utilizes a **Cyclic Directed Acyclic Graph (DAG)** to pit a "Researcher" against an "Adversary," ensuring that every compliance claim is cross-examined before human approval.

## 2. System Architecture

The system is built on three pillars:

1.  **Durable State:** Using LangGraph Checkpointers to allow audits to pause/resume.
2.  **Hybrid Retrieval:** Combining Semantic (Vector) and Deterministic (Knowledge Graph/ID-based) search.
3.  **Human-in-the-loop (HITL):** A mandatory "Break" in the graph for legal sign-off.

---

## 3. Detailed Project Structure

This structure follows the **Clean Architecture** principle, separating business logic from infrastructure.

```text
/ara-auditor
├── /data                       # Local persistent storage (Git ignored)
│   ├── /raw                    # Original Regulation PDFs
│   └── audit_checkpoints.db    # SQLite DB for LangGraph states
├── /src
│   ├── /api                    # FastAPI Layer
│   │   ├── main.py             # API Entry point
│   │   └── routes.py           # Audit execution & history endpoints
│   ├── /agents                 # The "Brains" (Logic Nodes)
│   │   ├── researcher.py       # RAG logic & Hybrid Search
│   │   ├── adversary.py        # Red-Teaming & Violation detection
│   │   └── auditor.py          # Summary & HITL preparation
│   ├── /core                   # System Backbone
│   │   ├── config.py           # Pydantic Settings & ENV management
│   │   ├── /graph
│   │   │   ├── state.py        # TypedDict State Definitions
│   │   │   └── workflow.py     # Compiled LangGraph Logic
│   ├── /services               # Infrastructure Layer (The "Hands")
│   │   ├── /vector             # Qdrant/Pinecone implementation
│   │   └── /db                 # Postgres/SQLite persistence
│   └── /prompts                # Version-controlled YAML prompts
│       ├── researcher.yaml
│       └── adversary.yaml
├── docker-compose.yml          # Local Qdrant, Redis, & Postgres
├── pyproject.toml              # Dependencies (LangGraph, Pydantic, etc.)
└── .env                        # API Keys (OPENAI_API_KEY, etc.)
```

---

## 4. Technical Workflow (Step-by-Step)

### Phase 1: Ingestion & Normalization

- **Action:** Documents are parsed via `LlamaParse`.
- **Senior Touch:** Burmese text is normalized to Unicode to prevent encoding mismatches in the Vector DB.

### Phase 2: The Research Node (`researcher.py`)

- **Logic:** Extracts keywords from the user's contract.
- **Retrieval:** Performs **Hybrid Search** (Dense + Sparse) in Qdrant. It retrieves laws by both meaning and specific Article IDs (e.g., "Directive 7/2024").

### Phase 3: The Adversary Node (`adversary.py`)

- **Logic:** Operates as a "Red Team." It receives the retrieved laws and the user's contract.
- **Goal:** Specifically looks for reasons to **reject** the contract. It calculates a `risk_score` based on found violations.

### Phase 4: The State Checkpoint & Breakpoint

- **Action:** The graph hits the `auditor` node.
- **Mechanism:** Because we use `interrupt_before=["auditor"]`, the graph "freezes." The current state (all found laws and violations) is serialized into `audit_checkpoints.db`.

### Phase 5: Human Approval

- **Action:** A human lawyer reviews the state via a dashboard or CLI.
- **Resolution:** The lawyer provides a `Command(resume=...)`. If approved, the graph completes. If rejected, it can be routed back to the `researcher` with human feedback.

---

## 5. Senior Engineering Features

| Feature                   | Implementation         | Why it's Senior Level                                |
| :------------------------ | :--------------------- | :--------------------------------------------------- |
| **Persistence**           | `SqliteSaver`          | Audits survive server crashes/restarts.              |
| **Adversarial Reasoning** | Multi-agent loop       | Prevents "Yes-man" AI bias by forcing a critique.    |
| **Hybrid RAG**            | Reciprocal Rank Fusion | High precision for specific legal IDs/Articles.      |
| **Traceability**          | LangSmith Integration  | Full observability into _which_ chunk caused a flag. |

---

## 6. Deployment Strategy

- **Inference:** Claude 3.5 Sonnet (for superior reasoning) or GPT-4o.
- **Database:** Qdrant (Vector) + PostgreSQL (Relational Audit Logs).
- **Orchestration:** LangGraph (Stateful Agent management).

---

## 7. Future Scalability (V3.0)

- **Temporal RAG:** Comparing a 2026 contract against 2024 laws to identify "compliance drift."
- **Knowledge Graph (Neo4j):** Mapping the relationships between different CBM directives to see which ones supersede others.

---
