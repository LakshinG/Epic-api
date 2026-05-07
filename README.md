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


# Local LLM Clinical Extraction Pipeline (REDCap / Epic FHIR)

## Overview
This repository contains a HIPAA-compliant, zero-cloud NLP pipeline designed to extract complex, unstructured clinical text into structured integer codes for the REDCap Epilepsy database. 

Due to the constraints of processing unredacted clinical notes and navigating IRB approvals, this architecture is designed to run **100% locally** on lab hardware (using Ollama and Llama/Qwen architectures). No patient data is ever transmitted to OpenAI, Anthropic, or external cloud providers.

## Core Features
* **Zero-Cloud Local Inference:** Utilizes `qwen2.5:14b` via Ollama for deterministic, secure clinical abstraction.
* **Strict Schema Adherence:** Leverages LangChain and Pydantic to enforce REDCap data dictionary rules, including mutually exclusive integers and multi-select arrays.
* **Batch Processing:** Automatically processes arrays of clinical progress notes and exports a `redcap_import_ready.csv` formatted exactly for REDCap bulk upload.
* **Epic FHIR Integration:** Includes backend API clients secured via RSA/JWT for querying structured patient demographics and medication data directly from Epic's FHIR endpoints.

## Architecture



1. **Input:** Unstructured clinical paragraphs (currently synthetic data for development).
2. **LLM Engine:** LangChain orchestration binds the REDCap Pydantic schema to the local 14B parameter model. Temperature is set to `0` for deterministic, repeatable extraction.
3. **Output:** A strict Python Dictionary/JSON object mapped to REDCap integer codes.
4. **Export:** Pandas aggregates the processed records and generates a batch CSV.

## Prerequisites
To run this pipeline locally, you will need:
* **Hardware:** A dedicated GPU with at least 8GB of VRAM (e.g., RTX 4070) is recommended to run the 14B model effectively.
* **Software:** Python 3.12+ and [Ollama](https://ollama.com/) installed and running in the background.

## Setup Instructions

**1. Clone the repository and activate the virtual environment:**
```powershell
git clone <your-repo-url>
cd Epic-api
.\venv\Scripts\Activate.ps1