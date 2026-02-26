import os
import json
from langchain.tools import tool
from typing import Optional, List, Dict
import datetime
from epic_fhir_client import EpicTokenClient

# Initialize the Epic Token Client using environment variable
# If token is not set, this will raise an error or create a client with None/empty string
# which will fail gracefully in the try/except blocks below.
token = os.environ.get("EPIC_TEST_TOKEN", "")
if not token:
    print("WARNING: EPIC_TEST_TOKEN environment variable not set. Real Epic integration will fail.")

client = EpicTokenClient(access_token=token)

@tool
def get_patient_tool(patient_id: str) -> str:
    """
    Retrieves the complete demographic profile for a patient.
    YOU MUST USE THIS TOOL to find emergency contacts, family members, 
    employers, phone numbers, addresses, and preferred languages.
    """
    try:
        # Fetch the data from Epic
        patient = client.get_patient(patient_id)

        # Return the ENTIRE JSON file to Gemini so it can read everything
        return json.dumps(patient)
        
    except Exception as e:
        return f"Error retrieving patient {patient_id}: {str(e)}"

@tool
def get_labs_tool(patient_id: str, loinc_code: Optional[str] = None) -> str:
    """
    Retrieves ALL laboratory test results for a patient. 
    Use this for labs, glucose, A1c, hemoglobin, or general 'results'.
    """
    try:
        bundle = client.get_observations(patient_id, category="laboratory")
        return json.dumps(bundle) # Hand the raw JSON bundle to Gemini
    except Exception as e:
        return f"Error retrieving labs: {str(e)}"

@tool
def get_medications_tool(patient_id: str) -> str:
    """
    Retrieves the complete medication list for a patient.
    Use this for 'medications', 'meds', 'prescriptions', or 'drugs'.
    """
    try:
        bundle = client.get_medications(patient_id)
        return json.dumps(bundle) # Hand the raw JSON bundle to Gemini
    except Exception as e:
        return f"Error retrieving medications: {str(e)}"

# Export the list of tools
tools = [get_patient_tool, get_labs_tool, get_medications_tool]