# Medical Data Agent Backend

A FastAPI-based backend that uses an AI agent (powered by LangChain and OpenAI) to interpret natural language queries from doctors and retrieve patient data from a mock Epic FHIR service.

## Features

- **Natural Language Understanding:** Uses OpenAI's GPT-4 (via LangChain) to understand queries like "Show me the last 3 A1c results".
- **Mock Epic FHIR Client:** Simulates an EHR system returning realistic "FHIR-Lite" JSON data for:
  - Patient Demographics
  - Lab Results (Observation)
  - Medications (MedicationRequest)
- **Tool Calling:** The agent intelligently calls specific Python functions (`get_patient`, `get_labs`, `get_medications`) based on the user's intent.
- **Transparent Data Sources:** The API returns both the natural language summary and the raw data used to generate the answer.

## Prerequisites

- Python 3.9+
- An OpenAI API Key

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your `OPENAI_API_KEY`.

## Usage

1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
   The server will start at `http://127.0.0.1:8000`.

2. Test the API:

   **Endpoint:** `POST /ask`

   **Example Request:**
   ```json
   {
     "doctor_query": "What is the patient's A1c trend?",
     "patient_id": "PT123"
   }
   ```

   **Example Curl:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"doctor_query": "Show me active medications", "patient_id": "PT123"}'
   ```

## Mock Data

The system is pre-loaded with mock data for testing:

- **Patient ID:** `PT123` (John Doe)
  - Has A1c, Glucose, and Hemoglobin labs.
  - Has Metformin and Lisinopril medications.
- **Patient ID:** `PT456` (Jane Smith)
  - Basic demographics only.

Any other Patient ID will result in empty data or a "Patient not found" error.

## Project Structure

- `main.py`: FastAPI application and API endpoint definition.
- `agent.py`: LangChain agent logic, prompt engineering, and tool execution loop.
- `tools.py`: Definitions of the tools the AI can use (decorated with `@tool`).
- `epic_service.py`: The mock client simulating Epic's FHIR API.
