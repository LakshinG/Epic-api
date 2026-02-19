from tools import get_patient_tool, get_labs_tool, get_medications_tool
import os

def test_real_integration():
    # Derrick Lin
    patient_id = "eq081-VQEgP8drUUqCWzHfw3"

    print(f"Testing Real Epic Sandbox Integration for Patient: {patient_id}")
    print(f"Token present: {'Yes' if os.environ.get('EPIC_TEST_TOKEN') else 'No'}")
    print("-" * 50)

    print("\n1. Testing Patient Demographics...")
    try:
        patient_info = get_patient_tool.invoke(patient_id)
        print(patient_info)
    except Exception as e:
        print(f"FAILED: {e}")

    print("\n2. Testing Labs...")
    try:
        # Get all labs
        labs = get_labs_tool.invoke(patient_id)
        print(f"--- All Labs (First 200 chars) ---\n{labs[:200]}...")
    except Exception as e:
        print(f"FAILED: {e}")

    print("\n3. Testing Medications...")
    try:
        meds = get_medications_tool.invoke(patient_id)
        print(f"--- Medications ---\n{meds}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_real_integration()
