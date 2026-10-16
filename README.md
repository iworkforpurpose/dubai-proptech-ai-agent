# Aman — Dubai Luxury PropTech AI Agent 🏙️✨

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash--Lite-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-8%2F8%20Passed%20(100%25)-10B981)](#-automated-testing--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)

An enterprise-grade autonomous real estate intelligence system and private wealth advisor for Dubai's luxury property market. Operating on Google's **Gemini 3.5 Flash-Lite** and **3,072-dimensional embeddings**, Aman delivers sub-second conversational search, strict mathematical filtering over **2,000 real Dubai properties**, interactive luxury property cards, autonomous VIP lead capture CRM, and instant financial amortization modeling.

---

## 🏛️ System Architecture

Aman replaces traditional brute-force vector search with a **Hybrid RAG** engine:

```
                          ┌──────────────────────────┐
                          │   Client Natural Query   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                      ┌────────────────────────────────────┐
                      │    Gemini 3.5 Flash-Lite Agent     │
                      │  - Parameter & Constraint Extractor│
                      │  - Multi-Step Autonomous Loop      │
                      │  - Session-Aware Memory Store      │
                      └──────────────┬───────────────┬─────┘
                                     │               │
                     Tool Call: search_dubai         Tool Call: book_viewing
                                     │               │
                                     ▼               ▼
                 ┌───────────────────────────┐ ┌───────────────────────────┐
                 │  PostgreSQL + pgvector    │ │  PostgreSQL CRM Table     │
                 │  - SQL WHERE Filters      │ │  - Lead Qualification     │
                 │  - Composite B-Tree Idxs  │ │  - Contact & Schedule     │
                 │  - 3,072-dim Cosine Rank  │ │  - Broker Alert System    │
                 └─────────────┬─────────────┘ └───────────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────────────────┐
        │  Saffron & Porcelain Luxury UI                          │
        │  - Advisory Markdown Dialogue                           │
        │  - Interactive DLD Verified Property Cards              │
        │  - Live Mortgage & Rental Yield Simulator               │
        │  - VIP Investor Desk Drawer                             │
        └─────────────────────────────────────────────────────────┘
```

---

## 💎 Key Features

- **2,000 Verified Real Dubai Properties**: Real-market data indexed with unique Property IDs (`DXB00001`–`DXB02000`), actual AED valuations, precise square footage, community amenities, and views (Palm Jumeirah, Downtown Dubai, JBR, Dubai Marina, Arabian Ranches, Mudon, The Greens).
- **Sub-5ms Hybrid Retrieval**: Hard financial and physical boundaries (budget ceilings, bedroom floors, location tags) are filtered in SQL using composite B-tree indexes, followed by 3,072-dimensional vector cosine distance ranking (`models/gemini-embedding-001` / `gemini-embedding-2`).
- **Autonomous Multi-Step Loop**: Dynamically searches multiple sub-categories sequentially when evaluating complex inquiries (e.g. comparing Downtown penthouses vs. Arabian Ranches villas).
- **Session-Aware Conversational Memory**: Retains multi-turn context without amnesia, answering analytical follow-up questions (*"Which of those has the lowest price per square foot?"* or *"Does that specific unit have a balcony?"*) without re-querying the database.
- **Autonomous Lead Qualification & CRM (BANT System)**: Captures qualified buyer intent directly through chat or via the interactive "Schedule VIP Viewing" modal, persisting leads to a dedicated `leads` schema.
- **Financial & Mortgage Simulator**: Interactive 25-year UAE debt amortization model (4.25% prime fixed rate, resident/non-resident LTV toggles) and estimated annual rental yield analysis (7.2% Dubai benchmark).
- **Zero-Cost Free-Tier Architecture**: Built entirely on Google AI Studio's free tier quotas with automatic model failover and rate-limit pacing.

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (for PostgreSQL + pgvector)
- **Google Gemini API Key** (obtain free from [Google AI Studio](https://aistudio.google.com/))

### 1. Clone & Environment Setup
```bash
git clone https://github.com/iworkforpurpose/dubai-proptech-ai-agent.git
cd dubai-proptech-ai-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql+psycopg2://admin:adminpassword@localhost:5433/proptech_db
```

### 3. Launch PostgreSQL + pgvector
Start the Docker database container:
```bash
docker compose up -d
```
*Note: Runs on local port `5433` with pgvector pre-installed.*

### 4. Ingest 2,000 Real Dubai Properties
Run the self-healing, rate-limit-protected ingestion pipeline:
```bash
./venv/bin/python scripts/ingest_real_data.py
```
*This downloads the real Dubai dataset, generates 3,072-dim embeddings via Google Gemini, creates composite B-tree indexes, and populates the database.*

### 5. Launch the FastAPI Application
```bash
uvicorn app.main:app --reload --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/chat` | Inquire with Aman. Returns synthesized advice + structured property cards. |
| `POST` | `/chat/reset` | Clears conversation history and cached properties for a session. |
| `POST` | `/api/leads` | Registers a VIP private viewing inquiry into the CRM database. |
| `GET` | `/api/leads` | Retrieves the latest captured leads for broker review. |
| `GET` | `/api/market-stats` | Returns real-time property count and market averages (AED price & ppsf). |

### Sample `/chat` Request & Response
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find me a 2-bedroom apartment in JBR under 2M AED with a sea view",
    "session_id": "client_session_1"
  }'
```
```json
{
  "reply": "Welcome. I am Aman, your private wealth real estate advisor...",
  "session_id": "client_session_1",
  "properties": [
    {
      "property_id": "DXB01426",
      "property_type": "Apartment",
      "location": "JBR",
      "bedrooms": 2,
      "bathrooms": 2,
      "size_sqft": 1371.0,
      "price_aed": 1957790.0,
      "price_per_sqft": 1428.0,
      "furnishing": "Furnished",
      "view_type": "Sea View",
      "amenities": "Swimming Pool, Gym, Jacuzzi, Balcony"
    }
  ]
}
```

---

## 🧪 Automated Testing & Verification

The repository includes a comprehensive automated test suite and two end-to-end edge-case harnesses.

### Run Unit & Integration Suite
```bash
PYTHONPATH=. ./venv/bin/pytest tests/test_production_suite.py -v
```
**Results (100% Pass Rate):**
- `test_db_connection_and_dataset` — Validates 2,000 real listings with 3,072-dim embeddings.
- `test_hybrid_sql_indexes` — Validates B-tree performance indexes in PostgreSQL.
- `test_search_dubai_properties_budget_constraint` — Validates strict budget math ceilings.
- `test_search_dubai_properties_bedrooms_constraint` — Validates bedroom count floors.
- `test_book_property_viewing_tool` — Validates lead CRM persistence.
- `test_api_market_stats` — Validates live benchmark endpoint.
- `test_api_leads_endpoints` — Validates lead creation and listing.
- `test_chat_endpoint_and_session_memory` — Validates multi-turn context retention.

### Run Edge-Case Test Harnesses
```bash
./venv/bin/python scripts/verify_all_edge_cases.py
./venv/bin/python scripts/verify_remaining_prompts.py
```
**Edge Cases Verified:**
1. **Hard Budget Ceilings**: Strict enforcement of budget limits with zero tolerance for over-budget properties.
2. **Colloquial Real Estate Syntax**: Natural extraction of `"2BHK"`, `"flat"`, `"Burj view"`, `"3.5M"`.
3. **Negative / Impossible Queries**: Gracefully declines impossible budgets (*"5-bed Palm villa for 500k AED"*) with market education rather than hallucinating fake data.
4. **Geographical Contradictions**: Detects impossible locations (*"Motor City beach penthouse"*) and recommends coastal alternatives.
5. **Multi-Turn Analytical Memory**: Evaluates lowest price/sqft and individual amenity checks across sequential turns without re-querying the database.
6. **Multi-Category Disjunctions**: Evaluates cross-community trade-offs (*"Downtown penthouse vs. Arabian Ranches villa"*).
7. **Lifestyle Semantic Vector Search**: Maps family intent (*"DIFC commute, 2 kids, quiet, parks, pool"*) to green communities.
8. **Autonomous Lead Booking**: Parses contact details from chat and inserts CRM appointments.
9. **Adversarial Jailbreak Guardrails**: Rebuffs prompt injections and protects system credentials.

---

## 🎨 Design Philosophy

Aman's interface is built on a bespoke luxury aesthetic adhering strictly to the **Anti-Generic Safe Harbor** protocol:
- **Zero-Purple Mandate**: No violet, indigo, or purple neon gradients.
- **Brand Palette**: Desert Saffron (`#D97706`), Obsidian Slate (`#0F172A`), and Warm Porcelain White (`#FAFAF8`).
- **Live Atmosphere**: Real-time Dubai GST Clock (`UTC+4`) and DLD Market Ticker integrated directly into the header.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
