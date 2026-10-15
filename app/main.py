import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Aman — Dubai Luxury PropTech AI Agent",
    description="Enterprise-grade autonomous real estate intelligence system for Dubai.",
    version="2.1.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ResetRequest(BaseModel):
    session_id: str


class LeadCreateRequest(BaseModel):
    full_name: str
    contact_info: str
    preferred_location: Optional[str] = None
    budget_max_aed: Optional[float] = None
    property_id: Optional[str] = None
    notes: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Accepts a natural language query and a session_id for multi-turn memory.
    Routes to the Gemini 3.5 Flash-Lite agent with Hybrid RAG search.
    Returns both conversational markdown advice and structured property cards.
    """
    from app.agent import chat_with_agent, get_session_properties

    reply = chat_with_agent(request.message, session_id=request.session_id)
    properties = get_session_properties(request.session_id)
    return {
        "reply": reply,
        "session_id": request.session_id,
        "properties": properties,
    }


@app.post("/chat/reset")
async def reset_session(request: ResetRequest):
    """Reset conversational memory and stored cards for a given session."""
    from app.agent import clear_session

    clear_session(request.session_id)
    return {"status": "success", "message": f"Session '{request.session_id}' cleared."}


@app.post("/api/leads")
async def create_lead(lead: LeadCreateRequest):
    """Register a VIP viewing request or consultation lead into the CRM."""
    from app.db.session import SessionLocal
    from app.db.models import Lead as LeadModel

    db = SessionLocal()
    try:
        new_lead = LeadModel(
            full_name=lead.full_name,
            contact_info=lead.contact_info,
            preferred_location=lead.preferred_location,
            budget_max_aed=lead.budget_max_aed,
            property_id=lead.property_id,
            notes=lead.notes,
            status="NEW_INQUIRY",
        )
        db.add(new_lead)
        db.commit()
        db.refresh(new_lead)
        return {
            "status": "success",
            "lead_id": new_lead.id,
            "message": "VIP viewing inquiry submitted. An Aman Private Wealth Advisor will contact you within 15 minutes.",
        }
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@app.get("/api/leads")
async def list_leads():
    """Retrieve the latest registered VIP client leads for broker review."""
    from app.db.session import SessionLocal
    from app.db.models import Lead as LeadModel

    db = SessionLocal()
    try:
        leads = (
            db.query(LeadModel).order_by(LeadModel.created_at.desc()).limit(50).all()
        )
        return [
            {
                "id": l.id,
                "full_name": l.full_name,
                "contact_info": l.contact_info,
                "property_id": l.property_id,
                "preferred_location": l.preferred_location,
                "budget_max_aed": l.budget_max_aed,
                "notes": l.notes,
                "status": l.status,
                "created_at": (
                    l.created_at.strftime("%Y-%m-%d %H:%M") if l.created_at else None
                ),
            }
            for l in leads
        ]
    finally:
        db.close()


@app.get("/api/market-stats")
async def market_stats():
    """Return live dataset counts and Dubai market benchmarks."""
    from app.db.session import SessionLocal
    from sqlalchemy import func
    from app.db.models import Property

    db = SessionLocal()
    try:
        total = db.query(Property).count()
        avg_price = db.query(func.avg(Property.price_aed)).scalar() or 0
        avg_ppsf = db.query(func.avg(Property.price_per_sqft)).scalar() or 0
        return {
            "total_properties": total,
            "average_price_aed": round(float(avg_price)),
            "average_price_per_sqft": round(float(avg_ppsf)),
            "market_sentiment": "Bullish (+4.2% YoY)",
            "currency": "AED",
        }
    finally:
        db.close()

