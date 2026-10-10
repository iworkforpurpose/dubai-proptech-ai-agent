from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from app.db.session import get_db
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Dubai Autonomous Real Estate Agent",
    description="API for the PropTech AI agent.",
    version="0.1.0"
)

# Mount the static directory for the frontend UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

# Load the SOTA embedding model into memory when the server starts
print("Loading BAAI/bge-large-en-v1.5 into API memory...")
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Takes a natural language query and passes it to the LangChain Agent (Llama 3.1).
    The Agent decides whether to respond conversationally or use the database tool.
    """
    from app.agent import chat_with_agent
    
    # Send user message to the LLM Agent
    agent_reply = chat_with_agent(request.message)
    
    return {
        "reply": agent_reply
    }
