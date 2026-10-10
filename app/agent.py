from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
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
def search_dubai_properties(query: str, max_results: int = 3) -> str:
    """
    Search the real estate database for properties in Dubai based on the user's natural language criteria.
    Use this tool whenever a user asks to find, buy, or rent a property, or mentions a location/budget.
    """
    db = SessionLocal()
    try:
        vector = embedding_model.encode(query).tolist()
        sql = text("""
            SELECT title, location, price_aed, property_type, bedrooms, bathrooms, amenities
            FROM properties
            ORDER BY embedding <=> :vector
            LIMIT :limit
        """)
        results = db.execute(sql, {"vector": str(vector), "limit": max_results}).fetchall()
        
        if not results:
            return "No matching properties found in the database."
            
        formatted_results = []
        for row in results:
            formatted_results.append(
                f"- {row.title} in {row.location}. {row.bedrooms} beds, {row.bathrooms} baths. "
                f"Price: {row.price_aed:,.0f} AED. Amenities: {row.amenities}."
            )
        return "\n".join(formatted_results)
    finally:
        db.close()

# Bind the tool to the model so it knows it can search the DB
llm_with_tools = llm.bind_tools([search_dubai_properties])

SYSTEM_PROMPT = """You are Aman, an elite, highly professional autonomous real estate agent in Dubai. 
You are speaking directly to a potential buyer or tenant.

RULES:
1. When a user asks for properties, you MUST use the `search_dubai_properties` tool.
2. NEVER invent or hallucinate properties. Only recommend properties returned by the tool.
3. Present the properties in a clean, luxurious, and readable format using Markdown.
   - Use **bold** for property titles and prices.
   - Use bullet points for amenities.
   - Keep your tone sophisticated and helpful, fitting for the Dubai luxury real estate market.
4. If a user just says 'hello', introduce yourself elegantly and ask what kind of property they are looking for in Dubai.
5. If no properties are found, politely suggest they adjust their criteria (e.g., budget or location).
"""

def chat_with_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # First pass: Let the LLM think and decide if it needs to use the database tool
    response = llm_with_tools.invoke(messages)
    
    # If the LLM decided to use the search tool
    if response.tool_calls:
        messages.append(response)
        
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_dubai_properties":
                # Execute the semantic search
                tool_msg = search_dubai_properties.invoke(tool_call)
                messages.append(tool_msg)
                
        # Second pass: Let the LLM read the database results and formulate a human response
        final_response = llm_with_tools.invoke(messages)
        return final_response.content
        
    # If it was just a greeting, return the conversational response
    return response.content
