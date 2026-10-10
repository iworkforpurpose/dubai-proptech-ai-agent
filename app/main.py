from fastapi import FastAPI

app = FastAPI(
    title="Aman - Autonomous Real Estate Agent",
    description="API for the PropTech WhatsApp and Voice AI agent.",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Aman PropTech Agent API is running."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
