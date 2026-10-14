"""
Production PropTech LangChain Agent — Powered by Gemini 2.0 Flash + Hybrid pgvector Search

Model: gemini-2.0-flash
  - Ultra-fast (<800ms) response latency
  - Native function calling and structured tool extraction
  - $0 cost on Google AI Studio Free Tier

Embeddings: text-embedding-004 (768-dim, centralized in app/embeddings.py)
Database: PostgreSQL + pgvector (Hybrid SQL constraints + cosine vector sorting)
Memory: Session-aware multi-turn conversation memory
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
from app.db.session import SessionLocal
from app.embeddings import get_embedding

# Retrieve API key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Initialize Gemini 2.0 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=api_key,
)

# In-memory session store: session_id -> list of messages
_session_store: dict[str, list] = {}


@tool
def search_dubai_properties(
    semantic_query: str,
    max_price_aed: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    property_type: Optional[str] = None,
    location: Optional[str] = None,
    furnishing: Optional[str] = None,
    view_type: Optional[str] = None,
    max_results: int = 4,
) -> str:
    """
    Search the real Dubai properties database using hybrid search (SQL constraints + vector ranking).

    Extract all applicable parameters from the client's query before calling:
      - semantic_query: descriptive phrase of the desired property (e.g. 'luxury waterfront penthouse with pool')
      - max_price_aed: budget ceiling in AED (e.g. '3 million' -> 3000000.0)
      - min_bedrooms: minimum bedroom count (e.g. '2BHK' -> 2)
      - property_type: 'Apartment', 'Villa', 'Townhouse', 'Penthouse', 'Studio'
      - location: specific neighbourhood (e.g. 'Downtown Dubai', 'Palm Jumeirah', 'JBR', 'Business Bay')
      - furnishing: 'Furnished', 'Unfurnished', 'Semi-Furnished'
      - view_type: 'Sea View', 'Burj Khalifa View', 'Marina View', 'City View'
      - max_results: number of listings to return (default 4)
    """
    db = SessionLocal()
    try:
        # Generate 768-dim embedding via Google text-embedding-004
        vector = get_embedding(semantic_query)

        clauses = ["WHERE 1=1"]
        params: dict = {"vector": str(vector), "limit": max_results}

        if max_price_aed is not None:
            clauses.append("AND price_aed <= :max_price")
            params["max_price"] = float(max_price_aed)
        if min_bedrooms is not None:
            clauses.append("AND bedrooms >= :min_beds")
            params["min_beds"] = int(min_bedrooms)
        if property_type:
            clauses.append("AND property_type ILIKE :ptype")
            params["ptype"] = f"%{property_type}%"
        if location:
            clauses.append("AND location ILIKE :loc")
            params["loc"] = f"%{location}%"
        if furnishing:
            clauses.append("AND furnishing ILIKE :furnish")
            params["furnish"] = f"%{furnishing}%"
        if view_type:
            clauses.append("AND view_type ILIKE :vtype")
            params["vtype"] = f"%{view_type}%"

        sql = text(f"""
            SELECT property_id, property_type, location, bedrooms, bathrooms,
                   size_sqft, price_aed, price_per_sqft, furnishing,
                   amenities, view_type, parking_spaces
            FROM properties
            {" ".join(clauses)}
            ORDER BY embedding <=> :vector
            LIMIT :limit
        """)

        results = db.execute(sql, params).fetchall()

        if not results:
            return (
                "No properties matched the exact filters. "
                "Suggest adjusting criteria such as budget, location, or bedrooms."
            )

        formatted = []
        for row in results:
            price_m = row.price_aed / 1_000_000
            ppsf = f"{row.price_per_sqft:,.0f}" if row.price_per_sqft else "N/A"
            formatted.append(
                f"**{row.property_type} — {row.location}** (`{row.property_id}`)\n"
                f"- **Price:** AED {price_m:.2f}M  |  AED {ppsf}/sqft\n"
                f"- **Details:** {row.bedrooms} Bed · {row.bathrooms} Bath · {row.size_sqft:,.0f} sqft\n"
                f"- **View & Finish:** {row.view_type} · {row.furnishing}\n"
                f"- **Parking:** {row.parking_spaces} reserved bay(s)\n"
                f"- **Amenities:** {row.amenities}\n"
            )

        return "\n---\n".join(formatted)

    finally:
        db.close()


# Bind search tool to Gemini 2.0 Flash
llm_with_tools = llm.bind_tools([search_dubai_properties])

SYSTEM_PROMPT = """You are Aman, an elite autonomous real estate advisor representing high-net-worth clients in Dubai's luxury property market.

TONE & DEMEANOR:
- Polished, authoritative, concise, and helpful. You speak with the precision of a senior private wealth advisor.
- Highlight AED pricing clearly, location prestige, and lifestyle amenities.

RULES:
1. When a client inquires about finding, buying, or viewing properties, ALWAYS invoke the `search_dubai_properties` tool.
2. Extract all explicit constraints accurately:
   - Budget -> max_price_aed (e.g. 'under 5 million' -> 5000000.0)
   - Beds -> min_bedrooms (e.g. '3 bedroom' -> 3)
   - Property type -> property_type (Apartment / Villa / Penthouse / Townhouse / Studio)
   - Neighbourhood -> location (e.g. 'Downtown Dubai', 'Palm Jumeirah', 'JBR')
   - View / Furnishing -> view_type, furnishing
3. CRITICAL: Whenever the tool returns property listings, you MUST present EVERY listing clearly in your reply. Never hide or summarize them into a single vague sentence.
4. Wrap the listings with a polite, sophisticated intro and offer to arrange a VIP private viewing.
5. If no properties match, suggest 1 or 2 specific adjustments (e.g. expanding to an adjacent community or adjusting budget).
6. On initial greetings ('hello', 'hi'), introduce yourself elegantly and ask what specific property requirements they have in Dubai.
7. Maintain awareness of the conversation history to provide seamless follow-ups."""


def chat_with_agent(user_input: str, session_id: str = "default") -> str:
    """
    Process user query with session memory and multi-step tool execution.
    """
    if session_id not in _session_store:
        _session_store[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]

    history = _session_store[session_id]
    history.append(HumanMessage(content=user_input))

    # First pass: Gemini plans and chooses whether to execute the tool
    response = llm_with_tools.invoke(history)
    history.append(response)

    # If Gemini invoked tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_dubai_properties":
                tool_result = search_dubai_properties.invoke(tool_call)
                history.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )

        # Second pass: Gemini reads the tool result and formats the final luxury response
        final_response = llm_with_tools.invoke(history)
        history.append(final_response)
        return final_response.content

    return response.content
