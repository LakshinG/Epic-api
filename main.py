from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Import the agent logic
# Note: This import expects OPENAI_API_KEY to be set in environment
try:
    from agent import process_query
except Exception as e:
    print(f"Warning: Could not import agent.process_query: {e}")
    # Define a dummy function to allow app startup for testing purposes
    def process_query(query: str, patient_id: str):
        raise HTTPException(status_code=500, detail="Agent not initialized. Check server logs.")

app = FastAPI(title="Medical Data Agent API")

class QueryRequest(BaseModel):
    doctor_query: str
    patient_id: str

class QueryResponse(BaseModel):
    response: str
    data_sources: List[Dict[str, Any]]

@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    """
    Endpoint to ask the medical agent a question about a specific patient.
    """
    try:
        # Call the agent
        result = process_query(request.doctor_query, request.patient_id)
        return result
    except Exception as e:
        # Log the error (print to console for now)
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
