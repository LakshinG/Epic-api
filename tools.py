import os
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
    Retrieves patient demographic information including name, date of birth, and gender.

    Args:
        patient_id (str): The ID of the patient (e.g., 'eq081-VQEgP8drUUqCWzHfw3').

    Returns:
        str: A simplified summary of patient details.
    """
    try:
        patient = client.get_patient(patient_id)

        # Parse relevant details to save token space
        name_list = patient.get("name", [])
        name_text = "Unknown"
        if name_list:
            # Try to build full name
            family = name_list[0].get("family", "")
            given = " ".join(name_list[0].get("given", []))
            name_text = f"{given} {family}".strip()

        dob = patient.get("birthDate", "Unknown")
        gender = patient.get("gender", "Unknown")

        return f"Patient Name: {name_text}\nDOB: {dob}\nGender: {gender}\nID: {patient.get('id')}"
    except Exception as e:
        return f"Error retrieving patient {patient_id}: {str(e)}"

@tool
def get_labs_tool(patient_id: str, loinc_code: Optional[str] = None) -> str:
    """
    Retrieves laboratory test results for a patient. Can be filtered by LOINC code.

    Use this tool when the user asks for specific lab results like 'Glucose', 'A1c', 'Hemoglobin',
    or generally for 'labs' or 'results'.

    Args:
        patient_id (str): The ID of the patient.
        loinc_code (Optional[str]): The LOINC code to filter results (e.g., '4548-4' for A1c).
                                    If not provided, returns all labs found in 'laboratory' category.

    Returns:
        str: A simplified list of observation results (Value, Unit, Date).
    """
    try:
        # Epic's API supports category='laboratory'
        bundle = client.get_observations(patient_id, category="laboratory")

        entries = bundle.get("entry", [])
        if not entries:
            return "No lab results found for this patient."

        results = []
        for entry in entries:
            resource = entry.get("resource", {})

            # Check LOINC filter if provided
            # Structure: resource -> code -> coding -> system/code
            codings = resource.get("code", {}).get("coding", [])
            if loinc_code:
                # If loinc_code is requested, skip if not present in any coding
                if not any(c.get("code") == loinc_code for c in codings):
                    continue

            # Extract display name
            display_name = resource.get("code", {}).get("text")
            if not display_name and codings:
                display_name = codings[0].get("display", "Unknown Test")

            # Extract Value
            # Structure: valueQuantity -> value, unit
            value_qty = resource.get("valueQuantity", {})
            value = value_qty.get("value")
            unit = value_qty.get("unit", "")

            # Extract Date
            effective_date = resource.get("effectiveDateTime", "Unknown Date")

            if value is not None:
                results.append(f"- {display_name}: {value} {unit} ({effective_date})")
            else:
                # Some labs might be qualitative (valueString or valueCodeableConcept)
                val_str = resource.get("valueString")
                if val_str:
                     results.append(f"- {display_name}: {val_str} ({effective_date})")

        if not results:
            if loinc_code:
                return f"No labs found matching LOINC code {loinc_code}."
            return "No lab results found."

        return "\n".join(results)

    except Exception as e:
        return f"Error retrieving labs: {str(e)}"

@tool
def get_medications_tool(patient_id: str, status: str = 'active') -> str:
    """
    Retrieves the list of medications for a patient, filtered by status (default is 'active').

    Use this tool when the user asks for 'medications', 'meds', 'prescriptions', or 'drugs'.

    Args:
        patient_id (str): The ID of the patient.
        status (str): The status of medications to retrieve (e.g., 'active', 'completed').

    Returns:
        str: A simplified list of medication requests (Name, Status, Date).
    """
    try:
        bundle = client.get_medications(patient_id)

        entries = bundle.get("entry", [])
        if not entries:
            return "No medication records found."

        med_list = []
        for entry in entries:
            resource = entry.get("resource", {})

            # Filter by status if requested
            # Note: real FHIR status might be 'active', 'stopped', 'completed'
            res_status = resource.get("status")
            if status and res_status != status:
                continue

            # Get Medication Name
            # medicationCodeableConcept -> text
            med_concept = resource.get("medicationCodeableConcept", {})
            med_name = med_concept.get("text")
            if not med_name and med_concept.get("coding"):
                med_name = med_concept["coding"][0].get("display", "Unknown Medication")

            authored_on = resource.get("authoredOn", "Unknown Date")

            med_list.append(f"- {med_name} (Status: {res_status}, Prescribed: {authored_on})")

        if not med_list:
            return f"No medications found with status '{status}'."

        return "\n".join(med_list)

    except Exception as e:
        return f"Error retrieving medications: {str(e)}"

# Export the list of tools
tools = [get_patient_tool, get_labs_tool, get_medications_tool]
