from langchain.tools import tool
from typing import Optional, List, Dict
from epic_service import MockEpicClient

# Initialize the client
client = MockEpicClient()

@tool
def get_patient_tool(patient_id: str) -> Dict:
    """
    Retrieves patient demographic information including name, date of birth, and gender.

    Args:
        patient_id (str): The ID of the patient (e.g., 'PT123').

    Returns:
        Dict: A dictionary containing patient details like resourceType, name, birthDate, and gender.
    """
    return client.get_patient(patient_id)

@tool
def get_labs_tool(patient_id: str, loinc_code: Optional[str] = None) -> List[Dict]:
    """
    Retrieves laboratory test results for a patient. Can be filtered by LOINC code.

    Use this tool when the user asks for specific lab results like 'Glucose', 'A1c', 'Hemoglobin',
    or generally for 'labs' or 'results'.

    Common LOINC codes:
    - HbA1c: '4548-4'
    - Glucose: '2345-7'
    - Hemoglobin: '718-7'

    Args:
        patient_id (str): The ID of the patient.
        loinc_code (Optional[str]): The LOINC code to filter results (e.g., '4548-4' for A1c).
                                    If not provided, returns all labs.

    Returns:
        List[Dict]: A list of observation resources containing values, units, and dates.
    """
    return client.get_labs(patient_id, loinc_code)

@tool
def get_medications_tool(patient_id: str, status: str = 'active') -> List[Dict]:
    """
    Retrieves the list of medications for a patient, filtered by status (default is 'active').

    Use this tool when the user asks for 'medications', 'meds', 'prescriptions', or 'drugs'.

    Args:
        patient_id (str): The ID of the patient.
        status (str): The status of medications to retrieve. Defaults to 'active'.
                      Other values could be 'completed', 'stopped', etc.

    Returns:
        List[Dict]: A list of medication request resources.
    """
    return client.get_medications(patient_id, status)

# Export the list of tools
tools = [get_patient_tool, get_labs_tool, get_medications_tool]
