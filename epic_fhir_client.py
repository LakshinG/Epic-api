"""
Epic FHIR API Client Library
============================
A comprehensive Python client for accessing Epic's FHIR API using their sandbox environment.

This module provides:
1. Backend/System authentication (JWT-based for server-to-server)
2. Open/Unsecured API access (for testing without OAuth)
3. Patient context authentication (SMART on FHIR OAuth2)

Epic Sandbox Information:
- Base URL: https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4
- Open API URL: https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4
- OAuth Token URL: https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token
- Test Patient Credentials: fhirjason / epicepic1
- Provider Credentials: USCDI / epicuscdi
"""

import requests
import uuid
import json
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


# ==================== CONFIGURATION ====================

@dataclass
class EpicConfig:
    """Epic FHIR API Configuration"""
    # Sandbox URLs
    FHIR_BASE_URL: str = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
    OAUTH_TOKEN_URL: str = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
    AUTHORIZE_URL: str = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/authorize"

    # Open (unsecured) API endpoints for testing
    OPEN_FHIR_BASE_URL: str = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"

    # Sandbox Test Patients (publicly available FHIR IDs from Epic documentation)
    TEST_PATIENTS: Dict[str, str] = None

    # Test Credentials (for patient-facing apps via MyChart)
    TEST_MYCHART_USER: str = "fhirjason"
    TEST_MYCHART_PASSWORD: str = "epicepic1"

    # Provider Test Credentials
    TEST_PROVIDER_USER: str = "USCDI"
    TEST_PROVIDER_PASSWORD: str = "epicuscdi"

    def __post_init__(self):
        if self.TEST_PATIENTS is None:
            # Common test patient FHIR IDs from Epic sandbox
            self.TEST_PATIENTS = {
                "Derrick Lin": "eq081-VQEgP8drUUqCWzHfw3",
                "Camila Lopez": "erXuFYUfucBZaryVksYEcMg3",
                "Jason Argonaut": "TzYvFYaURnsKKKMj.QPXnEhvEYPpL03oV4RnniSLigMcB",
                "Theodore Nelson": "Tbt3KuCY0B5PSrJvCu2j-PlK.aiHsu2xUjUM8bWpetXoB",
            }


# ==================== OPEN API CLIENT (No Auth Required) ====================

class EpicOpenClient:
    """
    Client for Epic's Open FHIR API endpoints.
    These endpoints don't require authentication and are perfect for initial testing.
    """

    def __init__(self, base_url: str = None):
        self.config = EpicConfig()
        self.base_url = base_url or self.config.OPEN_FHIR_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        })

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Read a specific patient by FHIR ID

        Args:
            patient_id: The patient's FHIR ID (e.g., "eq081-VQEgP8drUUqCWzHfw3")

        Returns:
            Patient FHIR resource as dictionary
        """
        url = f"{self.base_url}/Patient/{patient_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def search_patients(self, **params) -> Dict[str, Any]:
        """
        Search for patients with various parameters.

        Args:
            **params: Search parameters like family, given, birthdate, identifier

        Example:
            client.search_patients(family="Lopez", given="Camila")
        """
        url = f"{self.base_url}/Patient"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_conditions(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's conditions/diagnoses"""
        url = f"{self.base_url}/Condition"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_medications(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's medication requests"""
        url = f"{self.base_url}/MedicationRequest"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_observations(self, patient_id: str, category: str = None) -> Dict[str, Any]:
        """
        Get patient's observations (vitals, lab results, etc.)

        Args:
            patient_id: Patient FHIR ID
            category: Optional - 'vital-signs', 'laboratory', 'social-history', etc.
        """
        url = f"{self.base_url}/Observation"
        params = {"patient": patient_id}
        if category:
            params["category"] = category
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_allergies(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's allergy intolerances"""
        url = f"{self.base_url}/AllergyIntolerance"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_immunizations(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's immunization records"""
        url = f"{self.base_url}/Immunization"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_procedures(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's procedures"""
        url = f"{self.base_url}/Procedure"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_encounters(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's encounters"""
        url = f"{self.base_url}/Encounter"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_diagnostic_reports(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's diagnostic reports"""
        url = f"{self.base_url}/DiagnosticReport"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_documents(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's document references"""
        url = f"{self.base_url}/DocumentReference"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_clinical_notes(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient's clinical notes specifically.
        In Epic, these are often DocumentReference resources with category 'clinical-note'.
        """
        url = f"{self.base_url}/DocumentReference"
        params = {
            "patient": patient_id,
            "category": "clinical-note"
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_binary(self, binary_id: str) -> bytes:
        """
        Get raw binary content (e.g., for PDF notes).
        The binary_id usually comes from the DocumentReference content.attachment.url.
        """
        # If binary_id is a full URL, use it; otherwise construct it
        if binary_id.startswith("http"):
            url = binary_id
        else:
            url = f"{self.base_url}/Binary/{binary_id}"

        response = self.session.get(url, headers={"Accept": "application/pdf, application/json, text/plain, */*"})
        response.raise_for_status()
        return response.content

    def get_care_plans(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's care plans"""
        url = f"{self.base_url}/CarePlan"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_goals(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's goals"""
        url = f"{self.base_url}/Goal"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_practitioner(self, practitioner_id: str) -> Dict[str, Any]:
        """Get a practitioner by ID"""
        url = f"{self.base_url}/Practitioner/{practitioner_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_metadata(self) -> Dict[str, Any]:
        """Get server capability statement (metadata)"""
        url = f"{self.base_url}/metadata"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Get International Patient Summary (IPS) document
        Contains: problems, allergies, medications, immunizations
        """
        url = f"{self.base_url}/Patient/{patient_id}/$summary"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


# ==================== BACKEND SYSTEM CLIENT (JWT Auth) ====================

class EpicBackendClient:
    """
    Client for Epic's secured FHIR API using Backend System authentication.
    This uses JWT-based client_credentials flow for server-to-server communication.

    Requirements:
    1. Register an app at https://fhir.epic.com/
    2. Generate RSA key pair and upload public key
    3. Get your Non-Production Client ID
    """

    def __init__(
        self,
        client_id: str,
        private_key_path: str,
        base_url: str = None
    ):
        """
        Initialize the backend client.

        Args:
            client_id: Your Non-Production Client ID from Epic on FHIR
            private_key_path: Path to your RSA private key file (.pem)
            base_url: Optional custom base URL
        """
        self.config = EpicConfig()
        self.client_id = client_id
        self.base_url = base_url or self.config.FHIR_BASE_URL
        self.token_url = self.config.OAUTH_TOKEN_URL

        # Load private key
        with open(private_key_path, 'rb') as f:
            try:
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives.serialization import load_pem_private_key
                self.private_key = load_pem_private_key(f.read(), None, default_backend())
            except ImportError:
                print("Error: 'cryptography' library is required for EpicBackendClient.")
                print("Install it with: pip install cryptography")
                raise

        self.access_token = None
        self.token_expiry = None

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        })

    def _create_jwt(self) -> str:
        """Create a signed JWT for authentication"""
        now = datetime.now(tz=timezone.utc)
        payload = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.token_url,
            "jti": str(uuid.uuid4()),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=4)).timestamp()),
        }

        try:
            import jwt
        except ImportError:
            print("Error: 'PyJWT' library is required for EpicBackendClient.")
            print("Install it with: pip install PyJWT")
            raise

        token = jwt.encode(
            payload,
            self.private_key,
            algorithm="RS384",
            headers={"alg": "RS384", "typ": "JWT"}
        )
        return token

    def authenticate(self) -> str:
        """
        Obtain an access token using JWT assertion.

        Returns:
            Access token string
        """
        client_assertion = self._create_jwt()

        data = {
            "grant_type": "client_credentials",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": client_assertion
        }

        response = requests.post(self.token_url, data=data)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"Authentication Error: {e}")
            print(f"Response Body: {response.text}")
            raise

        token_data = response.json()
        print(f"  Granted Scopes: {token_data.get('scope', 'None')}")
        self.access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self.token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)

        # Update session headers
        self.session.headers["Authorization"] = f"Bearer {self.access_token}"

        return self.access_token

    def _ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self.access_token or datetime.now(tz=timezone.utc) >= self.token_expiry:
            self.authenticate()

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Read a patient by FHIR ID"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Patient/{patient_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def search_patients(self, **params) -> Dict[str, Any]:
        """Search for patients"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Patient"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_conditions(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's conditions/diagnoses"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Condition"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_medications(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's medication requests"""
        self._ensure_authenticated()
        url = f"{self.base_url}/MedicationRequest"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_observations(self, patient_id: str, category: str = None) -> Dict[str, Any]:
        """Get patient's observations"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Observation"
        params = {"patient": patient_id}
        if category:
            params["category"] = category
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_allergies(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's allergy intolerances"""
        self._ensure_authenticated()
        url = f"{self.base_url}/AllergyIntolerance"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_immunizations(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's immunization records"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Immunization"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_procedures(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's procedures"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Procedure"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_encounters(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's encounters"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Encounter"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_diagnostic_reports(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's diagnostic reports"""
        self._ensure_authenticated()
        url = f"{self.base_url}/DiagnosticReport"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_documents(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's document references"""
        self._ensure_authenticated()
        url = f"{self.base_url}/DocumentReference"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_clinical_notes(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's clinical notes"""
        self._ensure_authenticated()
        url = f"{self.base_url}/DocumentReference"
        params = {"patient": patient_id, "category": "clinical-note"}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_binary(self, binary_id_or_url: str) -> bytes:
        """Get raw binary content"""
        self._ensure_authenticated()
        if binary_id_or_url.startswith("http"):
            url = binary_id_or_url
        else:
            url = f"{self.base_url}/Binary/{binary_id_or_url}"
        response = self.session.get(url, headers={"Accept": "application/pdf, application/json, text/plain, */*"})
        response.raise_for_status()
        return response.content

    def get_care_plans(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's care plans"""
        self._ensure_authenticated()
        url = f"{self.base_url}/CarePlan"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_goals(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's goals"""
        self._ensure_authenticated()
        url = f"{self.base_url}/Goal"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def fhir_request(
        self,
        method: str,
        resource: str,
        resource_id: str = None,
        params: Dict = None,
        data: Dict = None
    ) -> Dict[str, Any]:
        """
        Make a generic FHIR API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            resource: FHIR resource type (Patient, Observation, etc.)
            resource_id: Optional resource ID
            params: Optional query parameters
            data: Optional request body
        """
        self._ensure_authenticated()

        url = f"{self.base_url}/{resource}"
        if resource_id:
            url = f"{url}/{resource_id}"

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()


# ==================== TOKEN-BASED CLIENT (For Bearer Tokens) ====================

class EpicTokenClient:
    """
    Client for Epic's FHIR API when you already have a bearer token.
    Useful for testing with tokens obtained through the web interface.
    """

    def __init__(self, access_token: str, base_url: str = None):
        """
        Initialize with an existing access token.

        Args:
            access_token: Bearer token (get from Epic sandbox web interface)
            base_url: Optional custom base URL
        """
        self.config = EpicConfig()
        self.base_url = base_url or self.config.FHIR_BASE_URL
        self.access_token = access_token

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
            "Authorization": f"Bearer {access_token}"
        })

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Read a patient by FHIR ID"""
        url = f"{self.base_url}/Patient/{patient_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def search_patients(self, **params) -> Dict[str, Any]:
        """Search for patients"""
        url = f"{self.base_url}/Patient"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_conditions(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's conditions/diagnoses"""
        url = f"{self.base_url}/Condition"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_medications(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's medication requests"""
        url = f"{self.base_url}/MedicationRequest"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_observations(self, patient_id: str, category: str = None) -> Dict[str, Any]:
        """Get patient's observations"""
        url = f"{self.base_url}/Observation"
        params = {"patient": patient_id}
        if category:
            params["category"] = category
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_allergies(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's allergy intolerances"""
        url = f"{self.base_url}/AllergyIntolerance"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_immunizations(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's immunization records"""
        url = f"{self.base_url}/Immunization"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_procedures(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's procedures"""
        url = f"{self.base_url}/Procedure"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_encounters(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's encounters"""
        url = f"{self.base_url}/Encounter"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_diagnostic_reports(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's diagnostic reports"""
        url = f"{self.base_url}/DiagnosticReport"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_documents(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's document references"""
        url = f"{self.base_url}/DocumentReference"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_clinical_notes(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's clinical notes"""
        url = f"{self.base_url}/DocumentReference"
        params = {"patient": patient_id, "category": "clinical-note"}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_binary(self, binary_id_or_url: str) -> bytes:
        """Get raw binary content"""
        if binary_id_or_url.startswith("http"):
            url = binary_id_or_url
        else:
            url = f"{self.base_url}/Binary/{binary_id_or_url}"
        response = self.session.get(url, headers={"Accept": "application/pdf, application/json, text/plain, */*"})
        response.raise_for_status()
        return response.content

    def get_care_plans(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's care plans"""
        url = f"{self.base_url}/CarePlan"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def get_goals(self, patient_id: str) -> Dict[str, Any]:
        """Get patient's goals"""
        url = f"{self.base_url}/Goal"
        response = self.session.get(url, params={"patient": patient_id})
        response.raise_for_status()
        return response.json()

    def fhir_request(
        self,
        method: str,
        resource: str,
        resource_id: str = None,
        params: Dict = None,
        data: Dict = None
    ) -> Dict[str, Any]:
        """Make a generic FHIR API request"""
        url = f"{self.base_url}/{resource}"
        if resource_id:
            url = f"{url}/{resource_id}"

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=data
        )
        response.raise_for_status()
        return response.json()


# ==================== HELPER FUNCTIONS ====================

def format_patient_info(patient: Dict) -> str:
    """Format patient FHIR resource into readable string"""
    output = []

    # Name
    if "name" in patient:
        for name in patient["name"]:
            full_name = []
            if "prefix" in name:
                full_name.extend(name["prefix"])
            if "given" in name:
                full_name.extend(name["given"])
            if "family" in name:
                full_name.append(name["family"])
            output.append(f"Name: {' '.join(full_name)}")

    # Birth date
    if "birthDate" in patient:
        output.append(f"Birth Date: {patient['birthDate']}")

    # Gender
    if "gender" in patient:
        output.append(f"Gender: {patient['gender']}")

    # Address
    if "address" in patient:
        for addr in patient["address"]:
            addr_parts = []
            if "line" in addr:
                addr_parts.extend(addr["line"])
            if "city" in addr:
                addr_parts.append(addr["city"])
            if "state" in addr:
                addr_parts.append(addr["state"])
            if "postalCode" in addr:
                addr_parts.append(addr["postalCode"])
            output.append(f"Address: {', '.join(addr_parts)}")

    # Phone
    if "telecom" in patient:
        for telecom in patient["telecom"]:
            if telecom.get("system") == "phone":
                output.append(f"Phone: {telecom.get('value')}")
            elif telecom.get("system") == "email":
                output.append(f"Email: {telecom.get('value')}")

    # Identifiers
    if "identifier" in patient:
        for ident in patient["identifier"]:
            id_type = ident.get("type", {}).get("text", "ID")
            output.append(f"{id_type}: {ident.get('value')}")

    return "\n".join(output)


def format_bundle_entries(bundle: Dict, resource_type: str = None) -> List[Dict]:
    """Extract entries from a FHIR Bundle response"""
    entries = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource_type is None or resource.get("resourceType") == resource_type:
            entries.append(resource)
    return entries


def print_response(response: Dict, indent: int = 2):
    """Pretty print a FHIR response"""
    print(json.dumps(response, indent=indent, default=str))


if __name__ == "__main__":
    # Quick test with open API
    print("Testing Epic FHIR API connection...")

    client = EpicOpenClient()

    # Test metadata endpoint
    try:
        metadata = client.get_metadata()
        print(f"✓ Connected to: {metadata.get('software', {}).get('name', 'Epic FHIR Server')}")
        print(f"  FHIR Version: {metadata.get('fhirVersion', 'Unknown')}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
