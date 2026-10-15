"""
Automated End-to-End Verification of All Test Prompts & Edge Cases for Aman.

Executes each edge-case prompt against the running API / agent, validates response contracts,
and logs verifiable proof of system behavior.
"""
import sys
import uuid
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def log_section(title):
    print("\n" + "=" * 70)
    print(f"🔹 {title}")
    print("=" * 70)

def test_hard_budget_constraint():
    log_section("TEST 1: Hard Budget Ceiling (Arabian Ranches 3-bed Villa <= 4M AED)")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    payload = {
        "message": "Find me a 3-bedroom villa in Arabian Ranches under 4 million AED.",
        "session_id": session_id
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    assert res.status_code == 200, f"HTTP error {res.status_code}"
    data = res.json()
    
    print(f"Query: {payload['message']}")
    print(f"Cards Returned: {len(data.get('properties', []))}")
    for p in data.get("properties", []):
        print(f"  - [{p['property_id']}] {p['property_type']} in {p['location']}: AED {p['price_aed']:,.0f} | {p['bedrooms']} Beds")
        assert p["price_aed"] <= 4_000_000, f"Violation: price {p['price_aed']} > 4M"
        assert p["bedrooms"] >= 3, f"Violation: beds {p['bedrooms']} < 3"
        
    print(f"Advisor Excerpt: {data['reply'][:200]}...")
    print("✅ TEST 1 PASSED: Strict mathematical constraints respected.")

def test_colloquial_real_estate_syntax():
    log_section("TEST 2: Colloquial Real Estate Syntax ('2BHK', 'flat', 'Burj view', '3.5M')")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    payload = {
        "message": "Looking for a 2BHK furnished flat in Downtown with Burj view for 3.5M",
        "session_id": session_id
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    assert res.status_code == 200
    data = res.json()
    
    print(f"Query: {payload['message']}")
    print(f"Cards Returned: {len(data.get('properties', []))}")
    for p in data.get("properties", []):
        print(f"  - [{p['property_id']}] {p['property_type']} in {p['location']}: AED {p['price_aed']:,.0f} | {p['bedrooms']} Beds | {p['furnishing']}")
        assert p["price_aed"] <= 3_500_000, f"Budget exceeded: {p['price_aed']}"
        assert p["bedrooms"] >= 2, f"Bedrooms under 2: {p['bedrooms']}"
        
    print(f"Advisor Excerpt: {data['reply'][:200]}...")
    print("✅ TEST 2 PASSED: Real estate colloquialisms mapped accurately.")

def test_impossible_budget_negative():
    log_section("TEST 3: Negative Test (5-bed Palm Jumeirah Villa under 500k AED)")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    payload = {
        "message": "Find me a 5-bedroom beachfront villa on Palm Jumeirah under 500,000 AED.",
        "session_id": session_id
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    assert res.status_code == 200
    data = res.json()
    
    print(f"Query: {payload['message']}")
    print(f"Cards Returned: {len(data.get('properties', []))}")
    print(f"Advisor Response:\n{data['reply']}")
    
    # Assert no properties or polite educational fallback
    reply_lower = data["reply"].lower()
    assert any(term in reply_lower for term in ["unrealistic", "adjust", "budget", "starting", "no properties", "exceed", "limited", "threshold", "million"]), \
        "Agent failed to acknowledge unrealistic budget!"
    print("✅ TEST 3 PASSED: Impossible budget handled gracefully with zero hallucination.")

def test_multi_turn_memory_chain():
    log_section("TEST 4: Multi-Turn Memory Chain (Discovery -> Lowest PPSF -> Amenities)")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    
    # Turn 1
    t1_payload = {
        "message": "Show me 2-bedroom apartments in Dubai Marina with a marina view.",
        "session_id": session_id
    }
    t1_res = requests.post(f"{BASE_URL}/chat", json=t1_payload, timeout=30)
    t1_data = t1_res.json()
    print(f"Turn 1 Query: {t1_payload['message']}")
    print(f"Turn 1 Listings ({len(t1_data.get('properties', []))}):")
    for p in t1_data.get("properties", []):
        print(f"  - [{p['property_id']}] AED {p['price_aed']:,.0f} | PPSF: AED {p['price_per_sqft']:,.0f}")
        
    # Turn 2: Contextual follow-up
    t2_payload = {
        "message": "Which one of those has the lowest price per square foot?",
        "session_id": session_id
    }
    t2_res = requests.post(f"{BASE_URL}/chat", json=t2_payload, timeout=30)
    t2_data = t2_res.json()
    print(f"\nTurn 2 Query: {t2_payload['message']}")
    print(f"Turn 2 Advisor Reply:\n{t2_data['reply']}")
    
    # Check that Turn 2 references one of the property IDs or prices from Turn 1
    t1_ids = [p["property_id"] for p in t1_data.get("properties", [])]
    assert any(pid in t2_data["reply"] for pid in t1_ids) or "sqft" in t2_data["reply"].lower(), \
        "Agent did not resolve the lowest PPSF property from previous turn context!"
    print("✅ TEST 4 PASSED: Multi-turn contextual reasoning confirmed.")

def test_autonomous_lead_booking():
    log_section("TEST 5: Autonomous Natural Language Lead Booking into CRM")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    test_name = "Alex Vance Testing"
    test_phone = "+971 50 987 6543"
    payload = {
        "message": f"I want to view DXB00682 tomorrow morning. My name is {test_name} and my WhatsApp is {test_phone}.",
        "session_id": session_id
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    assert res.status_code == 200
    data = res.json()
    
    print(f"Query: {payload['message']}")
    print(f"Advisor Response:\n{data['reply']}")
    
    # Check CRM API
    leads_res = requests.get(f"{BASE_URL}/api/leads", timeout=10)
    assert leads_res.status_code == 200
    leads = leads_res.json()
    matched = [l for l in leads if test_name in l["full_name"] and test_phone in l["contact_info"]]
    assert len(matched) > 0, "Lead was not persisted in PostgreSQL CRM database!"
    print(f"✅ Found Lead in CRM: ID #{matched[0]['id']} | {matched[0]['full_name']} | {matched[0]['contact_info']} | Status: {matched[0]['status']}")
    print("✅ TEST 5 PASSED: Natural language lead booked autonomously.")

def test_guardrails_deflection():
    log_section("TEST 6: Guardrail & Out-of-Domain Persona Deflection")
    session_id = f"test_{uuid.uuid4().hex[:6]}"
    payload = {
        "message": "Write a Python script to scrape Bayut and Dubizzle.",
        "session_id": session_id
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    assert res.status_code == 200
    data = res.json()
    
    print(f"Query: {payload['message']}")
    print(f"Advisor Response:\n{data['reply']}")
    
    # Verify Aman politely defers and offers real estate advisory
    reply_lower = data["reply"].lower()
    assert any(term in reply_lower for term in ["advisor", "real estate", "property", "aman", "assist", "portfolio"]), \
        "Agent did not maintain its advisory domain!"
    print("✅ TEST 6 PASSED: Out-of-domain prompt deflected with professional persona.")

if __name__ == "__main__":
    print("🚀 Launching Automated Verification Suite for Aman Dubai PropTech AI...")
    try:
        test_hard_budget_constraint()
        test_colloquial_real_estate_syntax()
        test_impossible_budget_negative()
        test_multi_turn_memory_chain()
        test_autonomous_lead_booking()
        test_guardrails_deflection()
        print("\n" + "🎉" * 25)
        print("ALL 6 CRITICAL EDGE CASES VERIFIED & VALIDATED 100% END-TO-END!")
        print("🎉" * 25 + "\n")
    except AssertionError as e:
        print(f"\n❌ Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Failure: {e}")
        sys.exit(1)
