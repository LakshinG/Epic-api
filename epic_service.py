from typing import Dict, List, Optional
import datetime

class MockEpicClient:
    """
    Simulates Epic's FHIR API endpoints, returning mock data.
    """

    def get_patient(self, patient_id: str) -> Dict:
        """
        Retrieves patient demographics.
        """
        # Mock logic based on ID.
        if patient_id == "PT123":
            return {
                "resourceType": "Patient",
                "id": patient_id,
                "name": [{"given": ["John"], "family": "Doe"}],
                "birthDate": "1980-01-01",
                "gender": "male"
            }
        elif patient_id == "PT456":
            return {
                "resourceType": "Patient",
                "id": patient_id,
                "name": [{"given": ["Jane"], "family": "Smith"}],
                "birthDate": "1992-05-15",
                "gender": "female"
            }
        else:
            return {
                "resourceType": "OperationOutcome",
                "issue": [{"severity": "error", "code": "not-found", "diagnostics": "Patient not found"}]
            }

    def get_labs(self, patient_id: str, loinc_code: Optional[str] = None) -> List[Dict]:
        """
        Retrieves lab results. If loinc_code is provided, filters by that code.
        """
        if patient_id != "PT123":
            return []

        today = datetime.date.today().isoformat()

        # Mock data for PT123
        mock_labs = [
            # A1c
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]}, "valueQuantity": {"value": 5.7, "unit": "%"}, "effectiveDateTime": "2023-10-12"},
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]}, "valueQuantity": {"value": 6.2, "unit": "%"}, "effectiveDateTime": "2023-05-10"},
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]}, "valueQuantity": {"value": 6.5, "unit": "%"}, "effectiveDateTime": "2023-01-15"},

            # Glucose
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "2345-7", "display": "Glucose"}]}, "valueQuantity": {"value": 95, "unit": "mg/dL"}, "effectiveDateTime": today},

            # Hemoglobin
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin"}]}, "valueQuantity": {"value": 14.5, "unit": "g/dL"}, "effectiveDateTime": "2023-08-20"}
        ]

        if loinc_code:
            filtered = [
                lab for lab in mock_labs
                if any(c.get('code') == loinc_code for c in lab.get('code', {}).get('coding', []))
            ]
            return filtered

        return mock_labs

    def get_medications(self, patient_id: str, status: str = 'active') -> List[Dict]:
        """
        Retrieves medication requests.
        """
        if patient_id != "PT123":
            return []

        # Mock data for PT123
        mock_meds = [
            {"resourceType": "MedicationRequest", "medicationCodeableConcept": {"text": "Metformin 500mg"}, "status": "active", "authoredOn": "2023-09-01"},
            {"resourceType": "MedicationRequest", "medicationCodeableConcept": {"text": "Lisinopril 10mg"}, "status": "active", "authoredOn": "2023-01-10"},
            {"resourceType": "MedicationRequest", "medicationCodeableConcept": {"text": "Amoxicillin 500mg"}, "status": "completed", "authoredOn": "2022-12-01"}
        ]

        if status:
            filtered = [med for med in mock_meds if med.get('status') == status]
            return filtered

        return mock_meds
