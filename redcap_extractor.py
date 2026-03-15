from pydantic import BaseModel
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json

# 1. Stripped Schema (LangChain handles the structure)
class REDCapEpilepsyData(BaseModel):
    sz_age: Optional[int]
    hand_dom: Optional[int]
    medhx_etio: Optional[int]
    medhx_prior_episgy: Optional[int]

# 2. Initialize the New 14B Model (Make sure the download finished!)
llm = ChatOllama(model="qwen2.5:14b", temperature=0)
structured_llm = llm.with_structured_output(REDCapEpilepsyData)

# 3. Explicit System Instructions (The AI reads this first)
system_instructions = """
You are an expert clinical data abstraction AI. Extract the requested fields from the clinical note.

CRITICAL REDCAP MAPPING RULES:
1. sz_age: The patient's age at FIRST seizure onset. (Do NOT use their current age).
2. hand_dom: Hand-dominance. Map to integer -> 1=Left, 2=Right, 3=Ambidextrous, 99=Other.
3. medhx_etio: Seizure type. Map to integer -> 0=Generalized, 1=Focal/Multifocal, 2=Both, 3=Psychogenic, 4=Physiologic.
4. medhx_prior_episgy: Did the patient have previous epilepsy surgery? Map to integer -> 1=Yes, 2=No.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{clinical_note}")
])

extraction_chain = prompt | structured_llm

# 4. Our Synthetic Test Note
synthetic_clinical_note = """
Patient is a 34-year-old right-handed female presenting to the UNC Epilepsy clinic for evaluation. 
Her first seizure occurred when she was 12 years old. Video EEG monitoring confirmed 
focal/multifocal epileptiform discharges. She reports no prior history of any brain 
surgeries or resections for her epilepsy.
"""

# 5. Run the Pipeline
print("Analyzing clinical note with explicit mapping...\n")
extracted_data = extraction_chain.invoke({"clinical_note": synthetic_clinical_note})

print("--- REDCAP STRUCTURED DATA ---")
if isinstance(extracted_data, dict):
    print(json.dumps(extracted_data, indent=2))
else:
    print(extracted_data.model_dump_json(indent=2))