from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional
import sys
import os

# Ensure the app context is loaded correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import SessionLocal
# We import the embedding model initialized in main
from app.main import embedding_model

# Initialize local LLM via Ollama
# We choose llama3.1 because it transfers perfectly to AWS Bedrock later for production
llm = ChatOllama(model="llama3.1", temperature=0)

@tool
def search_dubai_properties(
    semantic_query: str, 
    max_price_aed: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    location: Optional[str] = None,
    max_results: int = 4
) -> str:
    """
    Search the real estate database for properties in Dubai.
    Use this tool whenever a user asks to find, buy, or rent a property.
    Extract max_price_aed (e.g. 'under 3 million' = 3000000), min_bedrooms (e.g. '2BHK' = 2), and location if mentioned.
    The semantic_query should capture the vibe or specific amenities (e.g., 'luxurious waterfront apartment with pool').
    """
    db = SessionLocal()
    try:
        vector = embedding_model.encode(semantic_query).tolist()
        
        # Build dynamic SQL for Hybrid Search
        query_parts = [
            "SELECT title, location, price_aed, property_type, bedrooms, bathrooms, amenities",
            "FROM properties",
            "WHERE 1=1"
        ]
        params = {"vector": str(vector), "limit": max_results}
        
        if max_price_aed:
            query_parts.append("AND price_aed <= :max_price")
            params["max_price"] = max_price_aed
            
        if min_bedrooms:
            query_parts.append("AND bedrooms >= :min_beds")
            params["min_beds"] = min_bedrooms
            
        if location:
            query_parts.append("AND location ILIKE :loc")
            params["loc"] = f"%{location}%"
            
        query_parts.append("ORDER BY embedding <=> :vector LIMIT :limit")
        
        sql = text(" ".join(query_parts))
        results = db.execute(sql, params).fetchall()
        
        if not results:
            return "No matching properties found in the database. Tell the user to broaden their search criteria."
            
        formatted_results = []
        for row in results:
            # Format price nicely like 3.2M AED
            price_formatted = f"{(row.price_aed / 1000000):.2f}M" if row.price_aed >= 1000000 else f"{row.price_aed:,.0f}"
            formatted_results.append(
                f"- **{row.title} in {row.location}**\n"
                f"  - **Price:** {price_formatted} AED\n"
                f"  - **Size:** {row.bedrooms} Beds, {row.bathrooms} Baths\n"
                f"  - **Amenities:** {row.amenities}"
            )
        return "\n\n".join(formatted_results)
    finally:
        db.close()

# Bind the tool to the model so it knows it can search the DB
llm_with_tools = llm.bind_tools([search_dubai_properties])

SYSTEM_PROMPT = """You are Aman, an elite, highly professional autonomous real estate agent in Dubai. 
You are speaking directly to a potential luxury buyer or tenant.

RULES:
1. When a user asks for properties, you MUST use the `search_dubai_properties` tool.
2. Formulate the tool arguments carefully. Translate '2BHK' to min_bedrooms=2, '3 million' to max_price_aed=3000000.
3. NEVER invent or hallucinate properties. Only recommend the properties returned by the tool.
4. CRITICAL: The tool returns perfectly formatted Markdown lists of properties. You MUST include this exact markdown list in your final message to the user! Never hide the properties.
5. Wrap the property list with a luxurious, polite conversational greeting and closing.
6. If a user just says 'hello', introduce yourself elegantly and ask what kind of property they are looking for in Dubai.
"""

def chat_with_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        messages.append(response)
        
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_dubai_properties":
                tool_msg = search_dubai_properties.invoke(tool_call)
                messages.append(tool_msg)
                
        final_response = llm_with_tools.invoke(messages)
        return final_response.content
        
    return response.content
