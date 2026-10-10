from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from app.db.session import get_db

app = FastAPI(
    title="Dubai Autonomous Real Estate Agent",
    description="API for the PropTech AI agent.",
    version="0.1.0"
)

# Load the SOTA embedding model into memory when the server starts
print("Loading BAAI/bge-large-en-v1.5 into API memory...")
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
async def root():
    return {"message": "Dubai PropTech Agent API is running."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/search")
async def semantic_search(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Takes a natural language query from a buyer (e.g. 'I want a sea view villa under 5 million AED'),
    converts it to a vector, and uses pgvector to find the closest semantic matches.
    """
    # 1. Convert the natural language query into a mathematical vector
    query_vector = embedding_model.encode(request.query).tolist()
    
    # 2. Use pgvector's cosine distance operator (<=>) to find the most similar properties
    sql = text("""
        SELECT id, title, location, price_aed, property_type, bedrooms, bathrooms, amenities, description, 
               1 - (embedding <=> :vector) AS similarity_score
        FROM properties
        ORDER BY embedding <=> :vector
        LIMIT :top_k
    """)
    
    # Execute the raw SQL against the database
    results = db.execute(sql, {"vector": str(query_vector), "top_k": request.top_k}).fetchall()
    
    # 3. Format the results
    matched_properties = []
    for row in results:
        prop = dict(row._mapping)
        prop["similarity_score"] = round(prop["similarity_score"], 4)
        matched_properties.append(prop)
        
    return {
        "user_query": request.query,
        "results_found": len(matched_properties),
        "properties": matched_properties
    }
