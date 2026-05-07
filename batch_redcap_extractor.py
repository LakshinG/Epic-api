from pydantic import BaseModel
from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import pandas as pd
import json
import re

# 1. Expanded Pydantic Schema with Multi-Select Lists
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# 1. Expanded Pydantic Schema with Schema-Bound Constraints
class REDCapEpilepsyData(BaseModel):
    internal_clinical_reasoning: str = Field(
        description="Think step-by-step for EVERY field. Cite specific sentences from the text and explain why you chose each code before assigning it."
    )
    sz_age: Optional[int] = Field(
        description="The patient's age at FIRST seizure onset. Do not confuse with current age."
    )
    hand_dom: Optional[Literal[1, 2, 3, 99]] = Field(
        description="Hand-dominance."
    )
    medhx_etio: Optional[Literal[0, 1, 2, 3, 4]] = Field(
        description="Seizure type."
    )
    medhx_prior_episgy: Optional[Literal[1, 2]] = Field(
        description="Did the patient have PREVIOUS EPILEPSY SURGERY (like VNS, Lobectomy)? Look ONLY at the past surgical history. Having seizures or evaluating for surgery is NOT surgery."
    )
    demo_gender: Optional[Literal[1, 2, 3, 4, 99]] = Field(
        description="Patient Identified Gender."
    )
    demo_employed: Optional[Literal[1, 0, 999]] = Field(
        description="Employment status."
    )
    medhx_etio_focal: Optional[List[Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]]] = Field(
        description="Specific structural cause of Focal Seizures. If a physical cause like Tumor or TBI is NOT explicitly stated, you MUST map to 999. Do not confuse psychological triggers with etiology."
    )
    medhx_szsyndrome: Optional[Literal[1, 2]] = Field(
        description="Confirmed epilepsy syndrome presence."
    )
    medhx_priorepisgy_type: Optional[List[Literal[10, 11, 12, 13, 14, 999]]] = Field(
        description="Prior epilepsy surgeries."
    )
    medhx_neurohx: Optional[List[Literal[1, 2, 3, 4, 5, 0]]] = Field(
        description="Neurological Co-morbidities."
    )
    medhx_psych: Optional[List[Literal[1, 2, 3, 4, 5, 6, 7, 0, 999]]] = Field(
        description="Psychiatric Co-Morbidities."
    )

# 2. Initialize the 14B Model
llm = ChatOllama(model="qwen2.5:14b", 
                 temperature=0,
                 base_url="http://127.0.0.1:11434")
structured_llm = llm.with_structured_output(REDCapEpilepsyData)

# 3. Explicit System Instructions (Mapping Rules)
system_instructions = """
You are an expert clinical data abstraction AI. 
TASK: Extract REDCap variables from the clinical note into specific integer codes.

UNIVERSAL VERIFICATION RULES:
1. REASONING FIRST: You must evaluate EVERY single field in the `internal_clinical_reasoning` string before outputting any numbers. State the field name, cite the sentence from the text, and state the integer code you will use.
2. NEGATION CHECK: If a sentence contains "no history of", "denies", "negative for", or "not present", you MUST map that field to 0 or null.
3. CONTEXT CHECK: Ensure the diagnosis refers to the PATIENT, not family members.

CODE MAPPINGS (YOU MUST USE THESE EXACT INTEGERS):
- hand_dom: 1=Left, 2=Right, 3=Ambidextrous, 99=Other.
- medhx_etio: 0=Generalized, 1=Focal/Multifocal, 2=Both, 3=Psychogenic, 4=Physiologic.
- medhx_prior_episgy: 1=Yes, 2=No.
- demo_gender: 1=Male, 2=Female, 3=Transgender, 4=Non-binary, 99=Other.
- demo_employed: 1=Yes, 0=No, 999=Unknown.
- medhx_szsyndrome: 1=Yes, 2=No.
- medhx_etio_focal: 1=Mesial-temporal sclerosis, 2=Prior TBI, 3=Post-stroke/Vascular injury, 4=Post-infectious, 5=Tumor, 6=Vascular lesion, 7=Cortical Dysplasia, 8=Autoimmune, 9=Genetic, 10=Other Lesion, 999=Unknown.
- medhx_priorepisgy_type: 10=Multiple subpial transections, 11=Vagus nerve stimulation (VNS), 12=Deep brain stimulation (DBS), 13=Responsive neurostimulation (RNS), 14=Other, 999=Unknown.
- medhx_neurohx: 1=Stroke, 2=Hemorrhage, 3=TBI, 4=Dementia, 5=Headaches, 0=None. (Ignore negations).
- medhx_psych: 1=Depression, 2=Anxiety, 3=Bipolar Disorder, 4=PTSD, 5=Schizophrenia, 6=Alcohol/Substance Use, 7=Other, 0=None, 999=Unknown.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{clinical_note}")
])

extraction_chain = prompt | structured_llm

# # Mock notes for the sake of testing the extraction logic. In a real scenario, we would be reading from an uploaded file or database.
# synthetic_notes = [
#     """Patient is a 34-year-old right-handed female presenting to the clinic. She works full-time as an accountant. 
#     Her first seizure occurred when she was 12 years old. Video EEG monitoring confirmed focal epileptiform discharges 
#     caused by a prior traumatic brain injury (TBI). She has a history of severe migraines and headaches, as well as a prior ischemic stroke. 
#     She had two prior epilepsy surgeries: an anterior temporal lobectomy in 2018, and a Vagus nerve stimulator placed in 2021. 
#     She does not have a confirmed epilepsy syndrome.""",
    
#     """The patient is a 25-year-old male who is currently unemployed. Left-handed. 
#     Onset of generalized seizures at age 18. MRI shows no obvious lesions, etiology is likely genetic. 
#     He underwent a laser corpus callosotomy two years ago. Confirmed Juvenile myoclonic epilepsy syndrome.""",
    
#     """50-year-old female, ambidextrous. Seizures began recently at age 49 following an ischemic stroke. 
#     EEG shows focal discharges. No history of prior epilepsy surgeries. She is retired but works part-time at a local shop. Confirmed epilepsy syndrome is negative."""
# ]

# 4. Real Clinical Note Test
synthetic_notes = [
    """
    Assessment and Plan
    Principal Problem: Localization-related (focal) (partial) symptomatic epilepsy and epileptic syndromes with complex partial seizures, intractable, with status epilepticus (CMS-HCC)
    Active Problems: Anxious mood, Diabetes mellitus (CMS-HCC), Osteoporosis, Type 2 diabetes mellitus without complication (CMS-HCC), Essential hypertension, Pure hypercholesterolemia, unspecified

    Jane is a 66 y.o. female with a past medical history of frequent event who presents today for diagnostic evaluation to determine whether their typical events have an electrographic correlate.
    Increased seizure frequency: Due to frequent seizure-like events it is necessary to get a more definitive diagnosis of the seizure type.
    During this admission, we will consider using various activating procedures to provoke typical seizures including abrupt cessation of seizure medications, sleep deprivation, hyperventilation, photic stimulation, and exercise. We will aim to capture at least 3-5 typical seizures for epilepsy surgery evaluation. Strict seizure precautions and fall precautions with frequent vital sign and neuro checks will be used as these provoking features can put the patient at risk of worsening and more severe seizures such as status epilepticus. This type of test can only be safely performed in the inpatient setting due to the above safety concerns.

    Home medications : lamotrigine XR 300mg daily. Zonisamide 250mg nightly. Lorazepam 1mg PO rescue
    - Admit to EMU and Initiate video-EEG monitoring for the further characterization of the seizure and seizure like activity and diagnosis of specific type and localization of seizure.
    - Patient was made aware that he/ she may have to stay in the EMU for up to 5-7 days given that his/ her event only occurs about once a week. He/ She is informed that our goal is to capture 3 typical events for diagnostic confirmation. We will use activating procedures such as sleep deprivation, photic stimulation and hyperventilation and bedside bike pedaling. In addition, home anti-seizure medications will be reduced or stopped which may provoked more intense and frequent seizures compare to typical seizures experiencing at home.
    - For Anti-seizure medication management: hold zonisamide, Lamotrigine daily dose taken this am.
    - For other activating procedures no provoking procedures
    - Admission labs: CBC , CMP, Urine toxicology screen, anti-seizure medication levels pending
    Rescue Plan: 2 mg Ativan IV for convulsive seizures > 3 minutes or > /= 3 small events in 4 hrs. Confirm EEG correlate prior to treatment by consulting with EEG tech or page Dr.Waters.

    EMU Protocol
    -Initiate continuous video EEG monitoring
    -Initiate seizure precautions including padded siderails, cardiac telemetry, falls precautions, bed alarm, out of bed with assistance only, nursing seizure testing, and frequent vital signs and neuro check.
    -Insert and maintain a peripheral IV
    - In case of seizure, please follow EMU seizure protocol

    Following chronic stable medical conditions monitoring during this admission:
    HTN- losartan 50mg daily
    DM- metformin 1000mg twice a day. Ozempic 0.5mg weekly on Wednesday
    Hypercholesterolemia- atorvastatin 40mg daily
    Mood disorder- sertraline 100mg daily

    FEN/GI
    - replete lytes prn
    - regular diet

    PPx
    - DVT: Lovenox 40 mg Daily
    - GI: no indication for PPI at this time

    Access: PIV
    Code Status: full code. Confirmed on Admission.
    Contact Information
    PCP: SUSAN ALEXANDER, MD
    Primary Contact:
    DISPO: EMU, Neurology floor status
    Pending further characterization of seizure activity

    This patient was seen and discussed with Dr. Waters and Dr. Schmidt, Epilepsy fellow, who agree with the above assessment and plan.
    I directly provided 120 minutes of acute care time as documented in this note. Time includes: direct patient care, patient reassessment, coordination of patient care, interpretation of data, review of patient medical records, patient education, counseling and documentation of patient care. This time is exclusive of separately billable procedures.
    Linh Ngo, FNP
    UNCH EMU 123-7370

    HPI
    Chief Complaint: Seizure
    HPI: Ms. Jane is a 66 y.o. old right handed female has a past medical history of Depression, Diabetes mellitus (CMS-HCC), Hyperlipidemia, Hypertension, Neuropathy in diabetes (CMS-HCC), PTSD (post-traumatic stress disorder), and Seizures (CMS-HCC)., who is being admitted to the Epilepsy Monitoring Unit today for the further evaluation of seizures/spells.
    The patient is accompanied by her partner who contributes to the history. Patient has 2 types of seizure/seizure like events over the last 3 years. The patient reports to have unchanged seizure control. The seizure frequency is about every 2 months a month .

    Seizure type 1:sensory- onset since after the first seizure in 2019
    -Prodrome/Aura: feeling woozy, dizzy, very warm sensation through her body
    -Seizure Description: having to lay down. no nauseous, but vomiting the majority of the time. Full awareness and responsiveness
    -Duration: 2-3 minutes
    -Associated Symptoms: No tongue biting, urinary incontinence or post-ictal confusion
    Seizure Frequency: every 2 months with cluster of up to 5 event a day for 3-4 days
    Last seizure: end of Feb/March 2022, end of May, 2022, June xx 2022

    Seizure type 2: body stiffening- life time of 1 event in July 2019 prior to ASM
    -Prodrome/Aura: none
    -Seizure Description: LOC, body stiffening with generalized shaking, . Patient was amnesic to her event
    -Duration: unknown
    -Associated Symptoms: NoTongue biting, urinary incontinence, bowel incontinence. post-ictal confusion
    Seizure Frequency:once
    Last seizure: July 2019
    Seizure Onset Age: 63
    Seizure Triggers/ Provoking Features: stress
    -any significant life stress present / changes in stressful events- PTSD
    -The patient is currently not driving
    Previous Anti-seizure Drugs tried: Levetiracetam- mood issues

    Epilepsy Risk Factors:
    She was the product of an uncomplicated pregnancy and delivery. The patient had a normal development.There is no history febrile seizure as an infant or child, meningitis, encephalitis, known brain structural abnormality,or significant head trauma. There is no family history of seizures or epilepsy.

    Complications of seizures:
    Admission for status epilepticus: no
    Admission for frequent seizures: no
    Injuries during seizures or attributed to seizures: no

    Previous workup:
    EEG : 2EEGs at mission hospital that were normal
    vEEG/ EMU evaluation:
    MRI of brain: 8/2/2021- white matter disease
    Other studies:
    Most recent anti-seizure medication levels:

    Allergies
    No Known Allergies

    Current Medications
    No current facility-administered medications for this encounter.

    Past Medical History:
    Diagnosis,Date
    Depression
    Diabetes mellitus (CMS-HCC)
    Hyperlipidemia
    Hypertension
    Neuropathy in diabetes (CMS-HCC)
    PTSD (post-traumatic stress disorder)
    Seizures (CMS-HCC)

    Past Surgical History
    No past surgical history on file.

    Social History
    Tobacco Use
    Smoking status: Former Smoker
    Smokeless tobacco: Never Used
    Substance and Sexual Activity
    Alcohol use: Never
    Drug use: Never

    Family History
    History reviewed. No pertinent family history.
    Code Status: No Order

    Review of Systems
    A 12-system review of systems was conducted and was negative except as documented above in the HPI.
    Constitutional: Denies fever, or significant change in weight. Denies difficulty sleeping or excessive daytime sleepiness.
    HEENT: Denies hearing problems or sinus problems.
    Ophthalmologic: Denies vision problems.
    Cardiovascular: Denies palpitation, chest pain, or shortness of breath.
    Pulmonary: Denies cough, hemoptysis or asthma.
    GI: Denies abdominal pain, nausea, vomiting, or diarrhea.
    GU: Denies difficulty voiding.
    Endocrine: Denies history of diabetes or thyroid problems.
    Musculoskeletal: Denies joint or back pain.
    Psychiatric: Denies anxiety or depression.
    Skin: Denies significant rashes or lesions.
    Hematologic: Denies bleeding problems

    Objective
    Temp: [36.9 °C (98.4 °F)] 36.9 °C (98.4 °F)
    Heart Rate: [71] 71
    Resp: [16] 16
    BP: (135)/(73) 135/73
    MAP (mmHg): [92] 92
    SpO2: [96 %] 96 %
    No intake/output data recorded.

    Physical Exam:
    General Appearance:Well appearing. In no acute distress.
    HEENT: Head is atraumatic and normocephalic. Sclera anicteric without injection. Oropharyngeal membranes are moist with no erythema or exudate. Upper denture
    Neck: Supple.
    Lungs: Normal work of breathing. Clear to auscultation in anterior fields. No wheezes or crackles.
    Heart: Regular rate and rhythm. No murmurs, rubs, or gallops.
    Abdomen: Soft, nontender, nondistended. Bowl sounds are normal.
    Extremities: No clubbing, cyanosis, or edema.
    Psych: Appropriate affect and behavior
    Skin: No rash, lesions, or breakdown.

    Neurological Examination:
    Mental Status: Alert, conversant, able to follow conversation and interview. Spontaneous speech was fluent without word finding pauses, dysarthria, or paraphasic errors. Comprehension was intact. Memory for recent and remote events was intact. Memory: 3/3 registration, 3/3 recall at 3 minutes. Language was clear and fluent with intact naming, repetition, good fund of knowledge of current events, and good concentration.
    Cranial Nerves: PERRL. Pursuit eye movements were uninterrupted with full range and without end-gaze nystagmus. Facial sensation intact bilaterally to light touch in all three divisions of CNV. Face symmetric at rest. Normal facial movement bilaterally, including forehead, eye closure and grimace/smile. Hearing intact to conversation. Shoulder shrug full strength bilaterally. Palate movement is symmetric. Tongue protrudes midline and tongue movements are normal.
    Motor Exam: Normal bulk. No tremors, myoclonus, or other adventitious movement. Pronator drift is absent.
    Reflexes: DTRs are 2+ and symmetric throughout. Toes are downgoing bilaterally.
    Sensory: Sensation normal to light touch and temperature sensation to cold in both hands and both feet. decrease vibration in feet more right than left. Intact proprioception.
    Cerebellar/Coordination/Gait: Rapid alternating movements are normal in bilateral upper extremities. Finger-to-nose is normal without ataxia or dysmetria bilaterally. Heel-to-shin is normal without ataxia or dysmetria bilaterally. Gait exam demonstrates normal posture, base, stride length, arm swing and turns.

    Diagnostic Studies
    All Labs Last 24hrs:
    Recent Results
    No results found for this or any previous visit (from the past 24 hour(s)).
    """
]

# 4.5 THE DETERMINISTIC TEXT CHUNKER (Deprecated)
# def isolate_relevant_chunks(note_text):
#     """
#     A generalized negative-parser. Instead of guessing where the good data is,
#     it systematically deletes the highly-standardized 'noisy' sections of an H&P.
#     """
#     clean_text = note_text
#
#     # 1. Remove "Review of Systems" through to the next major section (usually Objective/Physical Exam)
#     clean_text = re.sub(r'Review of Systems.*?(?=Objective|Physical Exam|Diagnostic Studies|\Z)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
#
#     # 2. Remove "Objective / Physical Exam" through to Labs
#     clean_text = re.sub(r'(?:Objective|Physical Exam).*?(?=Diagnostic Studies|Labs|\Z)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
#
#     # 3. Remove "Diagnostic Studies / Labs" to the end
#     clean_text = re.sub(r'(?:Diagnostic Studies|All Labs).*?(?=\Z)', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
#
#     return clean_text.strip()

# 5. The Processing Loop
# 5. The Generalized Two-Pass Processing Loop
extracted_records = []

print("Starting extraction with multi-stage reasoning...\n")
for idx, note in enumerate(synthetic_notes):
    print(f"Analyzing Note #{idx + 1}...")
    try:
        # PASS 1: The "Distillation" Step
        # This removes 80% of the noise (physical exams, labs) that causes hallucinations.
        distill_prompt = (
            "Summarize the following clinical note into a concise medical profile. "
            "Focus ONLY on: Patient demographics, hand dominance, employment, "
            "detailed seizure history (onset age, syndrome, and etiology/focal types), "
            "past medical history (including specific neurological and psychiatric comorbidities), "
            "and all past surgical history (especially prior epilepsy surgeries like VNS or DBS). "
            "IGNORE: Physical exam findings, vital signs, and current lab results."
        )
        
        # We use the raw LLM to create a clean text summary first
        distilled_summary = llm.invoke(f"{distill_prompt}\n\n{note}")
        print("\n--- PASS 1: DISTILLED SUMMARY ---")
        print(distilled_summary.content)

        # PASS 2: The "Extraction" Step
        # Now we feed the CLEAN, short summary to the structured extractor.
        data = extraction_chain.invoke({"clinical_note": distilled_summary.content})
        
        if not isinstance(data, dict):
            data = data.model_dump()
            
        print("\n--- PASS 2: AI REASONING ---")
        print(data.get("internal_clinical_reasoning", "No reasoning provided."))
        print("\n------------------------------\n")

        data['record_id'] = idx + 1 
        extracted_records.append(data)
        
    except Exception as e:
        print(f"Error processing Note #{idx + 1}: {e}")

# 6. Export to CSV using Pandas
if extracted_records:
    df = pd.DataFrame(extracted_records)
    
    # --- NEW: REDCap Checkbox Expander ---
    def expand_checkboxes(df, column_name, possible_codes):
        for code in possible_codes:
            # Create REDCap formatted columns (e.g., medhx_neurohx___1)
            # Check if the code is in the AI's extracted list
            df[f"{column_name}___{code}"] = df[column_name].apply(
                lambda x: 1 if isinstance(x, list) and code in x else 0
            )
        # Drop the original list column so REDCap doesn't crash
        df = df.drop(columns=[column_name])
        return df

    # Expand the multi-select columns using the exact codes from your prompt
    if 'medhx_priorepisgy_type' in df.columns:
        df = expand_checkboxes(df, 'medhx_priorepisgy_type', [10, 11, 12, 13, 14, 999])
        
    if 'medhx_neurohx' in df.columns:
        df = expand_checkboxes(df, 'medhx_neurohx', [1, 2, 3, 4, 5, 0])

    if 'medhx_etio_focal' in df.columns:
        df = expand_checkboxes(df, 'medhx_etio_focal', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999])

    if 'medhx_psych' in df.columns:
        df = expand_checkboxes(df, 'medhx_psych', [1, 2, 3, 4, 5, 6, 7, 0, 999])
    # -------------------------------------

    # Drop the internal_clinical_reasoning column before export
    if 'internal_clinical_reasoning' in df.columns:
        df = df.drop(columns=['internal_clinical_reasoning'])
    
    # Reorder columns so record_id is first (REDCap requirement)
    cols = ['record_id'] + [col for col in df.columns if col != 'record_id']
    df = df[cols]
    
    export_filename = "redcap_import_ready.csv"
    df.to_csv(export_filename, index=False)
    print(f"\nExtraction complete! Saved {len(df)} records to {export_filename}")