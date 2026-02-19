#!/usr/bin/env python3
"""
Epic FHIR API - Usage Examples
==============================

This script demonstrates various ways to access Epic's FHIR sandbox:

1. Open API (no authentication required) - Great for testing
2. Token-based (when you have a bearer token)
3. Backend system (JWT authentication for server apps)

Epic Sandbox Credentials:
- MyChart Test User: fhirjason / epicepic1
- Provider Test User: USCDI / epicuscdi

Test Patient FHIR IDs:
- Derrick Lin: eq081-VQEgP8drUUqCWzHfw3
- Camila Lopez: erXuFYUfucBZaryVksYEcMg3
- Jason Argonaut: TzYvFYaURnsKKKMj.QPXnEhvEYPpL03oV4RnniSLigMcB
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from epic_fhir_client import (
    EpicOpenClient,
    EpicTokenClient,
    EpicConfig,
    format_patient_info,
    format_bundle_entries,
    print_response
)


def example_open_api():
    """
    Example 1: Using the Open API (No Authentication Required)
    
    This is the easiest way to get started with Epic FHIR sandbox.
    Perfect for initial testing and exploration.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Open FHIR API (No Auth Required)")
    print("="*60)
    
    # Initialize client
    client = EpicOpenClient()
    
    # Test patient FHIR ID from Epic sandbox
    # Use one of the publicly available test patients
    test_patient_id = "eq081-VQEgP8drUUqCWzHfw3"  # Derrick Lin
    
    print(f"\nFetching patient: {test_patient_id}")
    print("-" * 40)
    
    try:
        # 1. Get Patient Demographics
        print("\n1. Patient Demographics:")
        patient = client.get_patient(test_patient_id)
        print(format_patient_info(patient))
        
        # 2. Get Patient Conditions
        print("\n2. Conditions/Diagnoses:")
        conditions = client.get_conditions(test_patient_id)
        for entry in format_bundle_entries(conditions, "Condition"):
            code = entry.get("code", {}).get("text", "Unknown")
            status = entry.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "Unknown")
            print(f"  - {code} (Status: {status})")
        
        # 3. Get Medications
        print("\n3. Medications:")
        meds = client.get_medications(test_patient_id)
        for entry in format_bundle_entries(meds, "MedicationRequest"):
            med_text = entry.get("medicationCodeableConcept", {}).get("text", "Unknown")
            status = entry.get("status", "Unknown")
            print(f"  - {med_text} (Status: {status})")
        
        # 4. Get Vital Signs
        print("\n4. Vital Signs:")
        vitals = client.get_observations(test_patient_id, category="vital-signs")
        for entry in format_bundle_entries(vitals, "Observation"):
            obs_type = entry.get("code", {}).get("text", "Unknown")
            value = entry.get("valueQuantity", {})
            val_str = f"{value.get('value', '')} {value.get('unit', '')}"
            print(f"  - {obs_type}: {val_str}")
        
        # 5. Get Allergies
        print("\n5. Allergies:")
        allergies = client.get_allergies(test_patient_id)
        entries = format_bundle_entries(allergies, "AllergyIntolerance")
        if entries:
            for entry in entries:
                allergen = entry.get("code", {}).get("text", "Unknown")
                severity = entry.get("criticality", "Unknown")
                print(f"  - {allergen} (Severity: {severity})")
        else:
            print("  No allergies recorded")
        
        # 6. Get Immunizations
        print("\n6. Immunizations:")
        immunizations = client.get_immunizations(test_patient_id)
        for entry in format_bundle_entries(immunizations, "Immunization"):
            vaccine = entry.get("vaccineCode", {}).get("text", "Unknown")
            date = entry.get("occurrenceDateTime", "Unknown date")
            print(f"  - {vaccine} ({date})")
        
        # 7. Get Goals
        print("\n7. Health Goals:")
        goals = client.get_goals(test_patient_id)
        for entry in format_bundle_entries(goals, "Goal"):
            description = entry.get("description", {}).get("text", "Unknown")
            status = entry.get("lifecycleStatus", "Unknown")
            print(f"  - {description} (Status: {status})")
            
    except Exception as e:
        print(f"Error: {e}")


def example_search_patients():
    """
    Example 2: Searching for Patients
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Patient Search")
    print("="*60)
    
    client = EpicOpenClient()
    
    # Search by family name
    print("\nSearching for patients with family name 'Lopez'...")
    try:
        results = client.search_patients(family="Lopez")
        entries = format_bundle_entries(results, "Patient")
        print(f"Found {len(entries)} patient(s):")
        for patient in entries:
            print("\n" + format_patient_info(patient))
    except Exception as e:
        print(f"Search error: {e}")


def example_server_metadata():
    """
    Example 3: Getting Server Capabilities
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Server Metadata / Capability Statement")
    print("="*60)
    
    client = EpicOpenClient()
    
    try:
        metadata = client.get_metadata()
        
        print(f"\nServer: {metadata.get('software', {}).get('name', 'Unknown')}")
        print(f"FHIR Version: {metadata.get('fhirVersion', 'Unknown')}")
        print(f"Publisher: {metadata.get('publisher', 'Unknown')}")
        
        # List supported resources
        print("\nSupported Resources:")
        rest = metadata.get("rest", [{}])[0]
        resources = rest.get("resource", [])
        
        # Group by common categories
        clinical_resources = []
        admin_resources = []
        other_resources = []
        
        clinical_types = ["Patient", "Observation", "Condition", "Procedure", 
                        "MedicationRequest", "AllergyIntolerance", "Immunization",
                        "DiagnosticReport", "Encounter", "CarePlan"]
        admin_types = ["Practitioner", "Organization", "Location", "Schedule", "Slot"]
        
        for resource in resources:
            res_type = resource.get("type")
            if res_type in clinical_types:
                clinical_resources.append(res_type)
            elif res_type in admin_types:
                admin_resources.append(res_type)
            else:
                other_resources.append(res_type)
        
        print(f"\n  Clinical ({len(clinical_resources)}): {', '.join(sorted(clinical_resources))}")
        print(f"\n  Administrative ({len(admin_resources)}): {', '.join(sorted(admin_resources))}")
        print(f"\n  Other ({len(other_resources)}): {', '.join(sorted(other_resources[:10]))}...")
        
    except Exception as e:
        print(f"Error: {e}")


def example_patient_summary():
    """
    Example 4: Getting International Patient Summary (IPS)
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: International Patient Summary ($summary)")
    print("="*60)
    
    client = EpicOpenClient()
    test_patient_id = "eq081-VQEgP8drUUqCWzHfw3"  # Derrick Lin
    
    try:
        print(f"\nFetching IPS for patient: {test_patient_id}")
        summary = client.get_patient_summary(test_patient_id)
        
        print(f"\nDocument type: {summary.get('resourceType')}")
        print(f"Total entries: {summary.get('total', len(summary.get('entry', [])))}")
        
        # Parse the sections
        entries = summary.get("entry", [])
        resource_types = {}
        for entry in entries:
            res_type = entry.get("resource", {}).get("resourceType")
            resource_types[res_type] = resource_types.get(res_type, 0) + 1
        
        print("\nContents:")
        for res_type, count in sorted(resource_types.items()):
            print(f"  - {res_type}: {count}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_with_bearer_token():
    """
    Example 5: Using a Bearer Token (from Epic web interface)
    
    To get a bearer token:
    1. Go to https://fhir.epic.com/
    2. Register/Login
    3. Navigate to Sandbox Test Data
    4. Use "Try It" on any API - copy the bearer token from the request
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Token-Based Authentication")
    print("="*60)
    
    # You would get this token from the Epic sandbox web interface
    # This is a placeholder - real tokens expire after ~1 hour
    sample_token = "3d0f1316-60bf-4256-9350-e1ae848642d6"
    
    print("""
    To use token-based authentication:
    
    1. Visit https://fhir.epic.com/ and create an account
    2. Go to API Specifications
    3. Select any API and click "Try It"
    4. After authorization, copy the Bearer token from the request
    5. Use it like this:
    
    ```python
    from epic_fhir_client import EpicTokenClient
    
    client = EpicTokenClient(access_token="your-bearer-token")
    patient = client.get_patient("eq081-VQEgP8drUUqCWzHfw3")
    ```
    """)


def example_backend_auth():
    """
    Example 6: Backend System Authentication (JWT)
    
    For server-to-server applications that need unattended access.
    Requires app registration and key pair generation.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Backend System Authentication (JWT)")
    print("="*60)
    
    print("""
    For backend/server authentication, you need:
    
    1. Register your app at https://fhir.epic.com/
       - Create a "Backend System" app
       - Note your Non-Production Client ID
    
    2. Generate an RSA key pair:
       ```bash
       # Generate private key
       openssl genrsa -out private_key.pem 2048
       
       # Generate public key certificate
       openssl req -new -x509 -key private_key.pem -out public_key.pem -days 365
       ```
    
    3. Upload public key to your Epic app registration
    
    4. Use the client:
       ```python
       from epic_fhir_client import EpicBackendClient
       
       client = EpicBackendClient(
           client_id="your-non-prod-client-id",
           private_key_path="path/to/private_key.pem"
       )
       
       # Client handles authentication automatically
       patient = client.get_patient("eq081-VQEgP8drUUqCWzHfw3")
       ```
    """)


def list_test_patients():
    """Show available test patients"""
    print("\n" + "="*60)
    print("EPIC SANDBOX TEST PATIENTS")
    print("="*60)
    
    config = EpicConfig()
    print("\nAvailable test patients (FHIR IDs):")
    print("-" * 50)
    for name, fhir_id in config.TEST_PATIENTS.items():
        print(f"  {name}: {fhir_id}")
    
    print("\nMyChart Login Credentials (for patient-facing apps):")
    print(f"  Username: {config.TEST_MYCHART_USER}")
    print(f"  Password: {config.TEST_MYCHART_PASSWORD}")
    
    print("\nProvider Login Credentials (for provider-facing apps):")
    print(f"  Username: {config.TEST_PROVIDER_USER}")
    print(f"  Password: {config.TEST_PROVIDER_PASSWORD}")


def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   EPIC FHIR API EXAMPLES                      ║
║              Sandbox Testing & Development                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # List test patients
    list_test_patients()
    
    # Run examples
    example_open_api()
    example_search_patients()
    example_server_metadata()
    example_patient_summary()
    example_with_bearer_token()
    example_backend_auth()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
