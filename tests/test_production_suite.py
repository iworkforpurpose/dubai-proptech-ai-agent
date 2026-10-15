"""
Comprehensive Production Verification Suite for Aman — Dubai PropTech AI Agent.

Tests:
1. PostgreSQL database connection & real dataset volume
2. B-Tree hybrid filter indexes
3. Hybrid search accuracy (SQL mathematical constraints + vector ranking)
4. Structured property card extraction
5. VIP lead booking tool & database persistence
6. REST API endpoints (/chat, /chat/reset, /api/leads, /api/market-stats)
7. Multi-turn conversational session memory
"""
import pytest
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import text
from app.db.session import SessionLocal, engine
from app.db.models import Property, Lead
from app.main import app
from app.agent import (
    search_dubai_properties,
    book_property_viewing,
    chat_with_agent,
    get_session_properties,
    clear_session,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_db_connection_and_dataset():
    """Verify database connection and dataset size."""
    db = SessionLocal()
    try:
        count = db.query(Property).count()
        assert count >= 1000, f"Expected at least 1,000 properties, found {count}"
        
        # Verify sample real listing structure
        sample = db.query(Property).filter(Property.property_id.isnot(None)).first()
        assert sample is not None
        assert sample.property_id.startswith("DXB")
        assert sample.price_aed > 0
        assert sample.size_sqft > 0
        assert sample.embedding is not None
        assert len(sample.embedding) == 3072
    finally:
        db.close()


def test_hybrid_sql_indexes():
    """Verify performance B-tree indexes exist in PostgreSQL."""
    with engine.connect() as conn:
        indexes = conn.execute(text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'properties'
        """)).fetchall()
        index_names = [r[0] for r in indexes]
        assert "idx_properties_price" in index_names
        assert "idx_properties_bedrooms" in index_names
        assert "idx_properties_location" in index_names
        assert "idx_properties_type" in index_names


def test_search_dubai_properties_budget_constraint():
    """Verify hard mathematical budget ceiling in hybrid search."""
    max_price = 2_000_000.0
    res_str = search_dubai_properties.invoke({
        "semantic_query": "apartment with sea view",
        "max_price_aed": max_price,
        "max_results": 4,
    })
    
    assert res_str is not None
    assert "No properties matched" not in res_str
    
    # Check structured output
    cards = get_session_properties("default")
    assert len(cards) > 0
    for card in cards:
        assert card["price_aed"] <= max_price, f"Property exceeded budget: {card['price_aed']}"


def test_search_dubai_properties_bedrooms_constraint():
    """Verify bedroom floor in hybrid search."""
    min_beds = 3
    search_dubai_properties.invoke({
        "semantic_query": "spacious family home",
        "min_bedrooms": min_beds,
        "max_results": 4,
    })
    
    cards = get_session_properties("default")
    assert len(cards) > 0
    for card in cards:
        assert card["bedrooms"] >= min_beds, f"Property had fewer beds than {min_beds}: {card['bedrooms']}"


def test_book_property_viewing_tool():
    """Verify lead booking tool creates database record."""
    test_contact = f"test_{uuid.uuid4().hex[:6]}@dubai-investor.ae"
    result = book_property_viewing.invoke({
        "full_name": "Sheikh Mansoor Test",
        "contact_info": test_contact,
        "property_id": "DXB00682",
        "preferred_date_or_time": "Tomorrow 11 AM",
        "notes": "Interested in cash purchase"
    })
    
    assert "VIP viewing successfully scheduled" in result
    
    # Verify in DB
    db = SessionLocal()
    try:
        lead = db.query(Lead).filter(Lead.contact_info == test_contact).first()
        assert lead is not None
        assert lead.full_name == "Sheikh Mansoor Test"
        assert lead.property_id == "DXB00682"
        assert lead.status == "VIEWING_SCHEDULED"
    finally:
        db.close()


def test_api_market_stats(client):
    """Verify GET /api/market-stats returns live benchmark data."""
    response = client.get("/api/market-stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_properties" in data
    assert data["total_properties"] >= 1000
    assert "average_price_aed" in data
    assert "average_price_per_sqft" in data
    assert data["currency"] == "AED"


def test_api_leads_endpoints(client):
    """Verify POST and GET /api/leads."""
    email = f"lead_{uuid.uuid4().hex[:6]}@royalfam.ae"
    post_res = client.post("/api/leads", json={
        "full_name": "Tariq Royal Test",
        "contact_info": email,
        "property_id": "DXB00003",
        "preferred_location": "Palm Jumeirah",
        "budget_max_aed": 15000000.0,
        "notes": "Looking for private beachfront access"
    })
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"
    lead_id = post_res.json()["lead_id"]

    get_res = client.get("/api/leads")
    assert get_res.status_code == 200
    leads = get_res.json()
    assert any(l["id"] == lead_id for l in leads)


def test_chat_endpoint_and_session_memory(client):
    """Verify multi-turn coherence and structured card return in /chat."""
    session_id = f"test_sess_{uuid.uuid4().hex[:8]}"
    
    # Turn 1: Search inquiry
    turn1_res = client.post("/chat", json={
        "message": "I am looking for a 2-bedroom apartment in JBR under 2.5 million AED with a sea view",
        "session_id": session_id,
    })
    assert turn1_res.status_code == 200
    turn1_data = turn1_res.json()
    assert "reply" in turn1_data
    assert len(turn1_data["reply"]) > 50
    assert "properties" in turn1_data
    assert len(turn1_data["properties"]) > 0
    assert turn1_data["properties"][0]["location"] is not None

    # Turn 2: Contextual follow-up
    turn2_res = client.post("/chat", json={
        "message": "What amenities did you say this property has?",
        "session_id": session_id,
    })
    assert turn2_res.status_code == 200
    turn2_data = turn2_res.json()
    assert "reply" in turn2_data
    # Aman answers about the amenities from conversation context
    assert any(w in turn2_data["reply"].lower() for w in ["amenities", "pool", "gym", "jacuzzi", "residence", "feature"])

    # Reset session
    reset_res = client.post("/chat/reset", json={"session_id": session_id})
    assert reset_res.status_code == 200
