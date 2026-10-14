"""
Upgraded LangChain Agent — GPT-4o + Hybrid Search + Session Memory

Model choice: GPT-4o
  - Best structured tool calling of any available model
  - Never drops property list from tool results
  - Correctly parses constraint arguments (price, bedrooms, view type)
  - Produces luxurious, well-structured Markdown output

Embedding choice: text-embedding-3-large (centralized in app/embeddings.py)
  - 3072-dim, highest accuracy available via API
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

from sqlalchemy import text
from app.db.session import SessionLocal
from app.embeddings import get_embedding

# Initialize GPT-4o — the unambiguous best choice for tool-calling accuracy
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# In-memory session store: session_id -> list of messages
# In production on AWS, this moves to DynamoDB or Redis
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
    Search the real Dubai properties database using hybrid search.

    Always extract these parameters from the user's message before calling:
      - semantic_query: a descriptive phrase of what they want (e.g. 'luxury apartment with pool and gym')
      - max_price_aed: budget ceiling (e.g. '3 million AED' → 3000000)
      - min_bedrooms: minimum bedroom count (e.g. '2BHK' or '2 bed' → 2)
      - property_type: 'Apartment', 'Villa', 'Townhouse', 'Penthouse', 'Studio'
      - location: neighbourhood name (e.g. 'Downtown Dubai', 'Palm Jumeirah', 'JBR')
      - furnishing: 'Furnished', 'Unfurnished', 'Semi-Furnished'
      - view_type: 'Sea View', 'Burj Khalifa View', 'City View', 'Marina View', 'Garden View'
    """
    db = SessionLocal()
    try:
        vector = get_embedding(semantic_query)

        clauses = ["WHERE 1=1"]
        params: dict = {"vector": str(vector), "limit": max_results}

        if max_price_aed is not None:
            clauses.append("AND price_aed <= :max_price")
            params["max_price"] = max_price_aed
        if min_bedrooms is not None:
            clauses.append("AND bedrooms >= :min_beds")
            params["min_beds"] = min_bedrooms
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
                "No properties matched the exact criteria. "
                "Suggest the user broaden their budget, location, or bedroom count."
            )

        formatted = []
        for row in results:
            price_m = row.price_aed / 1_000_000
            ppsf = f"{row.price_per_sqft:,.0f}" if row.price_per_sqft else "N/A"
            formatted.append(
                f"**{row.property_type} — {row.location}** `{row.property_id}`\n"
                f"- **Price:** AED {price_m:.2f}M  |  AED {ppsf}/sqft\n"
                f"- **Size:** {row.bedrooms} Bed · {row.bathrooms} Bath · {row.size_sqft:,.0f} sqft\n"
                f"- **View:** {row.view_type}  |  **Furnishing:** {row.furnishing}\n"
                f"- **Parking:** {row.parking_spaces} space(s)\n"
                f"- **Amenities:** {row.amenities}\n"
            )

        return "\n---\n".join(formatted)

    finally:
        db.close()


llm_with_tools = llm.bind_tools([search_dubai_properties])

SYSTEM_PROMPT = """You are Aman, an elite autonomous real estate agent specialising exclusively in Dubai's luxury property market.

PERSONALITY: Sophisticated, concise, and deeply knowledgeable. You speak like a senior private banker — not a call-center agent.

RULES:
1. When a user asks about properties, ALWAYS call the `search_dubai_properties` tool. No exceptions.
2. Extract ALL applicable filter parameters before calling the tool:
   - Budget → max_price_aed (e.g. "3 million" → 3000000)
   - Bedrooms → min_bedrooms (e.g. "2BHK" → 2)
   - Property type → property_type (Apartment / Villa / Penthouse / Townhouse / Studio)
   - Location → location (exact neighbourhood name)
   - Furnishing → furnishing (Furnished / Unfurnished / Semi-Furnished)
   - View → view_type (Sea View / Burj Khalifa View / City View / Marina View)
3. ALWAYS include the FULL formatted property list returned by the tool in your reply. Never summarise or omit listings.
4. Wrap the listing block in a brief, elegant introduction and a professional closing offer (e.g. schedule a viewing).
5. If no properties are found, suggest 1–2 ways to broaden the search.
6. On greeting, introduce yourself briefly and ask what the client is looking for — do not offer unsolicited listings.
7. You have persistent memory of this conversation. Use it to avoid repeating questions."""


def chat_with_agent(user_input: str, session_id: str = "default") -> str:
    """
    Process a user message within a named session.
    Session history is kept in-memory for multi-turn coherence.
    """
    if session_id not in _session_store:
        _session_store[session_id] = [SystemMessage(content=SYSTEM_PROMPT)]

    history = _session_store[session_id]
    history.append(HumanMessage(content=user_input))

    # First pass: LLM decides whether to call a tool or respond directly
    response = llm_with_tools.invoke(history)
    history.append(response)

    # If the LLM chose to use the database search tool
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_dubai_properties":
                tool_result = search_dubai_properties.invoke(tool_call)
                history.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )

        # Second pass: LLM reads the tool result and produces the final response
        final_response = llm_with_tools.invoke(history)
        history.append(final_response)
        return final_response.content

    return response.content
