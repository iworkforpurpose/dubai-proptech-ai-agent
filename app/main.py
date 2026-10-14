import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="Aman — Dubai PropTech AI Agent",
    description="Production-grade real estate AI agent for Dubai.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Accepts a natural language query and a session_id for multi-turn memory.
    Routes to the GPT-4o agent with Hybrid RAG search.
    """
    from app.agent import chat_with_agent

    reply = chat_with_agent(request.message, session_id=request.session_id)
    return {"reply": reply, "session_id": request.session_id}
