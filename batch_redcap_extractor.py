from pydantic import BaseModel
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import pandas as pd
import json

# 1. Expanded Pydantic Schema
class REDCapEpilepsyData(BaseModel):
    sz_age: Optional[int]
    hand_dom: Optional[int]
    medhx_etio: Optional[int]
    medhx_prior_episgy: Optional[int]
    demo_gender: Optional[int]
    demo_employed: Optional[int]
    medhx_etio_focal: Optional[int]
    medhx_szsyndrome: Optional[int]

# 2. Initialize the 14B Model
llm = ChatOllama(model="qwen2.5:14b", temperature=0)
structured_llm = llm.with_structured_output(REDCapEpilepsyData)

# 3. Explicit System Instructions (Mapping Rules)
system_instructions = """
You are an expert clinical data abstraction AI. Extract the requested fields from the clinical note.
If a field is not explicitly mentioned, leave it as null.

CRITICAL REDCAP MAPPING RULES:
1. sz_age: The patient's age at FIRST seizure onset.
2. hand_dom: Hand-dominance. Map to -> 1=Left, 2=Right, 3=Ambidextrous, 99=Other.
3. medhx_etio: Seizure type. Map to -> 0=Generalized, 1=Focal/Multifocal, 2=Both, 3=Psychogenic, 4=Physiologic.
4. medhx_prior_episgy: Did the patient have previous epilepsy surgery? Map to -> 1=Yes, 2=No.
5. demo_gender: Patient Identified Gender. Map to -> 1=Male, 2=Female, 3=Transgender, 4=Non-binary, 99=Other.
6. demo_employed: Employed. Map to -> 1=Yes, 0=No, 999=Unknown.
7. medhx_etio_focal: Etiology of Seizure. Map to -> 1=Mesial-temporal sclerosis, 2=Prior TBI, 3=Post-stroke, 5=Tumor, 9=Genetic.
8. medhx_szsyndrome: Confirmed epilepsy syndrome? Map to -> 1=Yes, 2=No.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{clinical_note}")
])

extraction_chain = prompt | structured_llm

# 4. Synthetic Batch of Doctor's Notes
synthetic_notes = [
    """Patient is a 34-year-old right-handed female presenting to the clinic. She works full-time as an accountant. 
    Her first seizure occurred when she was 12 years old. Video EEG monitoring confirmed focal epileptiform discharges 
    caused by a prior traumatic brain injury (TBI). She reports no prior history of any brain surgeries. She does not have a confirmed epilepsy syndrome.""",
    
    """The patient is a 25-year-old male who is currently unemployed. Left-handed. 
    Onset of generalized seizures at age 18. MRI shows no obvious lesions, etiology is likely genetic. 
    He underwent a laser corpus callosotomy two years ago. Confirmed Juvenile myoclonic epilepsy syndrome.""",
    
    """50-year-old female, ambidextrous. Seizures began recently at age 49 following an ischemic stroke. 
    EEG shows focal discharges. No history of prior epilepsy surgeries. She is retired but works part-time at a local shop. Confirmed epilepsy syndrome is negative."""
]

# 5. The Processing Loop
extracted_records = []

print("Starting bulk clinical extraction...\n")
for idx, note in enumerate(synthetic_notes):
    print(f"Analyzing Note #{idx + 1}...")
    try:
        data = extraction_chain.invoke({"clinical_note": note})
        
        # Convert Pydantic object to dictionary if it isn't one already
        if not isinstance(data, dict):
            data = data.model_dump()
            
        data['record_id'] = idx + 1 # Add a REDCap Record ID
        extracted_records.append(data)
    except Exception as e:
        print(f"Error processing Note #{idx + 1}: {e}")

# 6. Export to CSV using Pandas
if extracted_records:
    df = pd.DataFrame(extracted_records)
    
    # Reorder columns so record_id is first (REDCap requirement)
    cols = ['record_id'] + [col for col in df.columns if col != 'record_id']
    df = df[cols]
    
    export_filename = "redcap_import_ready.csv"
    df.to_csv(export_filename, index=False)
    print(f"\nExtraction complete! Saved {len(df)} records to {export_filename}")
    print("\nPreview of extracted data:")
    print(df.head())