import json
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import re

class VariableReasoning(BaseModel):
    variable_name: str = Field(description="The EXACT name of the REDCap field from the schema.")
    evidence_quote: str = Field(description="Exact sentence from the summary proving your choice.")
    chosen_code: str = Field(description="The final output. IF A COUNT OR SINGLE CHOICE, output ONLY the single integer (e.g., '2'). IF AN ARRAY, output a comma-separated list of the integer codes (e.g., '2, 8').")

# --- 1. PYDANTIC SCHEMAS ---
# --- 1. PYDANTIC SCHEMAS ---
class HistoryExtraction(BaseModel):
    step_by_step_logic: List[VariableReasoning] = Field(description="MANDATORY: You must create a reasoning entry for EVERY History variable.")
    
    # Metadata
    patient_id: str = Field(description="The Patient ID from the metadata.")
    document_reference_id: str = Field(description="The Document Reference ID from the metadata.")
    note_date: str = Field(description="The exact date of the note from the metadata (YYYY-MM-DD).")
    title: str = Field(description="The exact title of the note from the metadata.")

    # Demographics & General History
    sz_age: int = Field(description="Age at FIRST seizure onset. CRITICAL: DO NOT use current demographic age.")
    hand_dom: Optional[int] = Field(description="Hand-dominance.")
    demo_gender: Optional[int] = Field(description="Patient Identified Gender.")
    demo_employed: Optional[int] = Field(description="Employment status.")
    medhx_driving: Optional[int] = Field(description="Is this patient currently driving?")
    medhx_si: Optional[int] = Field(description="Known history of Suicidal Ideation/Suicide Attempt?")

    # Epilepsy & Seizure Details
    medhx_etio: Optional[int] = Field(description="Seizure type.")
    medhx_etio_focal: Optional[List[int]] = Field(description="Specific structural cause of Focal Seizures.")
    emu_epilepsytype: Optional[int] = Field(description="Epilepsy Type.")
    medhx_szsyndrome: Optional[int] = Field(description="Confirmed epilepsy syndrome presence.")
    medhx_szsyndrome_type: Optional[int] = Field(description="If syndrome present, specific epilepsy syndrome.")
    emu_sz_type: Optional[int] = Field(description="Please select most frequent seizure type.")
    emu_sz_type1_freq: Optional[int] = Field(description="Frequency of Primary Seizure Type.")
    emu_epilepsy_intract: Optional[int] = Field(description="Is the patient's epilepsy medically refractory?")

    # Surgery & Comorbidities
    medhx_prior_episgy: int = Field(description="PREVIOUS NEUROSURGERY for epilepsy? 1=Yes, 2=No.")
    medhx_priorepisgy_type: Optional[List[int]] = Field(description="Prior epilepsy surgeries.")
    medhx_sgy_cand_yn: Optional[int] = Field(description="Is this patient a surgical candidate? (Historical)")
    emu_sxcandidate: Optional[int] = Field(description="Surgery Candidate / Is patient a surgery candidate? (Current Plan)")
    medhx_neurohx: Optional[List[int]] = Field(description="Neurological Co-morbidities.")
    medhx_psych: Optional[List[int]] = Field(description="Psychiatric Co-Morbidities.")
    
    # Discharge
    emu_dcevents_type: Optional[int] = Field(description="Discharge Diagnosis.")

class MedicationExtraction(BaseModel):
    step_by_step_logic: List[VariableReasoning] = Field(description="MANDATORY: You must create a reasoning entry for EVERY Medication variable.")
    emu_asm_number: int = Field(description="Total number of ASMs the patient was taking AT ADMISSION.")
    emu_asm_type: List[int] = Field(description="HOME / ADMISSION MEDICATIONS.")
    emu_asm_sfx: Optional[int] = Field(description="Was patient experiencing side effects from ASMS?")
    emu_asmdc_number: int = Field(description="Total number of ASMs prescribed AT DISCHARGE.")
    emu_asmdc_type: List[int] = Field(description="DISCHARGE MEDICATIONS.")

class ImagingExtraction(BaseModel):
    step_by_step_logic: List[VariableReasoning] = Field(description="MANDATORY: You must create a reasoning entry for EVERY Imaging variable.")
    mri_yn: Optional[int] = Field(description="Was an MRI performed? 1=Yes, 2=No but ordered, 0=No.")
    mri_normal_abnormal: Optional[int] = Field(description="If MRI performed, was it Normal or Abnormal?")
    mri_lateralization: Optional[int] = Field(description="Lateralization of MRI findings.")
    mri_lesion_left: Optional[int] = Field(description="Left hemisphere MRI lesion types.")
    mri_lesion_right: Optional[int] = Field(description="Right hemisphere MRI lesion types.")
    mri_l_localization: Optional[int] = Field(description="Left hemisphere MRI localization.")
    mri_r_localization: Optional[int] = Field(description="Right hemisphere MRI localization.")
    pet_yn: Optional[int] = Field(description="Was FDG-PET performed?")
    fmri_yn: Optional[int] = Field(description="Was fMRI performed?")
    wada_yn: Optional[int] = Field(description="Was WADA performed?")

# --- 2. INITIALIZE 32B MODEL ---
llm = ChatOllama(
    model="qwen2.5:32b", 
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

# --- 3. SYSTEM INSTRUCTIONS ---
history_system_instructions = """
You are an expert clinical data abstraction AI. 
TASK: Extract Demographics, Metadata, and History REDCap variables.

UNIVERSAL VERIFICATION RULES:
1. REASONING FIRST: Evaluate EVERY field in `step_by_step_logic` first.
2. NEGATION CHECK: If a sentence contains "no history of" or "denies", map that field to the specific 'No' code (e.g., 0 or 2).

DEMOGRAPHICS & SOCIAL MAPPINGS:
- sz_age: Age at FIRST seizure onset. ABSOLUTE RULE: You must use the explicitly stated "Seizure Onset Age" (e.g., 63). DO NOT use the patient's current demographic age.
- hand_dom: 1=Left, 2=Right, 3=Ambidextrous, 99=Other.
- demo_gender: 1=Male, 2=Female, 3=Transgender, 4=Non-binary, 99=Other.
- demo_employed: 1=Yes, 0=No, 999=Unknown.
- medhx_driving: 1=Yes, 2=No, 3=Unknown.
- medhx_si (Suicidal Ideation): 1=Yes, 0=No.

EPILEPSY MAPPINGS:
- medhx_etio: 1=Focal/Multifocal, 2=Generalized, 3=Unknown. (If note says "focal" or "localization-related", MUST code 1).
- medhx_etio_focal: Cause of Focal Seizures. 1=MTS... 10=Other, 999=Unknown. DEPENDENCY RULE: If medhx_etio is 1, and no specific structural cause is explicitly stated, you MUST output the array [999].
- emu_epilepsytype: 1=Focal-Single focus, 2=Focal-Two foci, 3=Multifocal(3+), 4=Generalized-idiopathic, 5=Generalized-symptomatic, 6=Unlocalizable. DEPENDENCY RULE: If diagnosis is "Localization-related (focal)", default to 1 (Focal-Single focus). Do not use 6 unless explicitly stated as unlocalizable.
- medhx_szsyndrome: Confirmed syndrome. 1=Yes, 2=No. 
- medhx_szsyndrome_type: 1=MTLE-HS... 15=Genetic NOS, 999=Other. DEPENDENCY RULE: If medhx_szsyndrome is 2 (No), medhx_szsyndrome_type MUST be left empty/null.
- emu_sz_type (Most frequent): 1=PGTC, 2=Focal motor with retained awareness, 3=Focal non-motor with retained awareness, 4=Focal motor impaired awareness, 5=Focal non-motor impaired awareness, 6=Events retained awareness-NOS, 7=Staring spells-NOS, 8=Hypermotor-NOS, 9=Myoclonus, 10=Convulsions NOS, 99=Other. CRITICAL RULE: If the note explicitly states "sensory-onset" with "full awareness", you MUST bypass code 3 and select 6 (Events retained awareness-NOS) to match the legacy database schema.
- emu_sz_type1_freq: 1=Multi/day, 2=Daily... 9=Random clusters, 99=Other. (CRITICAL RULE: If the note mentions seizures occurring in "clusters", you MUST output 9).
- emu_epilepsy_intract (Refractory): 1=Yes, 2=No, 3=Unclear.

SURGERY & COMORBIDITIES:
- medhx_prior_episgy: 1=Yes, 2=No. Output 2 if no history.
- medhx_priorepisgy_type: 10=MST, 11=VNS... 999=Unknown. DEPENDENCY RULE: If medhx_prior_episgy is 2 (No), you MUST output an empty array [].
- medhx_sgy_cand_yn (History of candidacy): 1=Yes, 2=No, 3=Unclear. 
- emu_sxcandidate (Current candidacy): 1=Yes/discussed, 2=Yes/not amenable, 3=Yes/not discussed, 4=Possible future, 5=No, 999=Unknown. 
- CRITICAL SURGERY RULE: If the patient is admitted for "epilepsy surgery evaluation", you MUST output 1 for BOTH medhx_sgy_cand_yn and emu_sxcandidate.
- medhx_neurohx: 1=Stroke, 2=Hemorrhage, 3=TBI, 4=Dementia, 5=Headaches/Neuropathy, 0=None. CRITICAL RULE: If "Neuropathy" is listed in the past medical history, you MUST include 5.
- medhx_psych: 1=Depression, 2=Anxiety, 3=Bipolar, 4=PTSD, 5=Schizophrenia, 6=Alcohol/Substance, 7=Other, 0=None, 999=Unknown.

DISCHARGE:
- emu_dcevents_type: 1=Epilepsy, 2=FND, 3=Mixed FND/Epilepsy, 4=Physiologic Non-epileptic, 5=Inconclusive, 6=Other.
- CRITICAL GUARDRAIL: If the document is an "Admission Note", a discharge diagnosis does not exist yet. Leave emu_dcevents_type blank/null.
"""

meds_system_instructions = """
You are an expert clinical pharmacist abstracting REDCap data. 
TASK: Extract Admission and Discharge Anti-Seizure Medications.

STEPHEN RULE: Differentiate between Home/Admission and Discharge meds. 
- Admission Meds (emu_asm_type): Include all home ASMs, EVEN IF held/paused.
- Discharge Meds (emu_asmdc_type): Only include meds actively prescribed at the END of the hospital stay. 
- CRITICAL GUARDRAIL: If the document is an "Admission Note", "H&P", or "Admit to EMU", discharge medications DO NOT EXIST YET. You are STRICTLY FORBIDDEN from extracting any discharge meds. You MUST output emu_asmdc_number as 0 and emu_asmdc_type as an empty array [].

SIDE EFFECTS:
- emu_asm_sfx (Experiencing side effects?): 1=Yes, 2=No, 99=Unclear.

ABSOLUTE EXCLUSION RULE:
You must strictly EXCLUDE any medication labeled as PRN, "rescue", or "as needed". Do NOT count them. 

MEDICATION LEGEND:
1=levetiracetam, 2=lamotrigine, 3=carbamazepine, 4=oxcarbazepine, 5=eslicarbazepine, 6=brivaracetam, 7=topiramate, 8=zonisamide, 9=clobazam, 10=clonazepam, 11=diazepam, 12=lorazepam, 13=valproic acid, 14=gabapentin, 15=lacosamide, 16=pregabalin, 17=phenytoin, 18=phenobarbital, 19=cannabidiol, 20=cenobamate, 21=ethosuximide, 22=rufinamide, 23=felbamate, 24=perampanel, 25=acetazolamide, 26=primidone, 27=stiripentol, 28=vigabatrin, 29=fenfluramine, 99=Other
"""

imaging_system_instructions = """
You are an expert clinical neuroradiologist abstracting REDCap data.
TASK: Extract Neuroimaging and diagnostic studies.

IMAGING LEGEND & RULES:
- mri_yn, pet_yn, fmri_yn, wada_yn: 1=Yes, 2=No but ordered, 0=No. 
- mri_normal_abnormal: 1=Normal, 2=Abnormal.
- mri_lateralization: 1=Left, 2=Right, 3=Bilateral, 4=Midline, 5=Multifocal. 

MRI LESIONS (mri_lesion_left, mri_lesion_right):
1=Hippocampal sclerosis/MTS, 2=Cavernoma/vascular, 3=Neoplasm/glioma, 4=FCD, 5=Stroke/Encephalomalacia/TBI, 6=Polymicrogyria, 7=Heterotopias, 8=Cortical Tubers/TSC, 0=Multiple, 99=Other. 
DEPENDENCY RULE: If a lesion is not explicitly lateralized to the Left or Right in the text, you MUST leave the left/right lesion arrays empty []. Do not guess.

MRI LOCALIZATION (mri_l_localization, mri_r_localization):
1=Temporal, 2=Frontal, 3=Parietal, 4=Occipital, 5=Sub-cortical, 6=Other, 999=None of the above.

DEPENDENCY RULE: If mri_yn is 0 (No), all other MRI variables MUST be left empty/null.
"""

# --- 4. BUILD CHAINS ---
history_chain = ChatPromptTemplate.from_messages([("system", history_system_instructions), ("human", "{clinical_note}")]) | llm.with_structured_output(HistoryExtraction)
meds_chain = ChatPromptTemplate.from_messages([("system", meds_system_instructions), ("human", "{clinical_note}")]) | llm.with_structured_output(MedicationExtraction)
imaging_chain = ChatPromptTemplate.from_messages([("system", imaging_system_instructions), ("human", "{clinical_note}")]) | llm.with_structured_output(ImagingExtraction)

synthetic_notes = [
    """
    Metadata:
    record_id: 1
    patient_id: real-001
    document_reference_id: real-doc-001
    note_date: 2026-04-09
    title: EMU Admission Note - Focal Epilepsy Unknown Etiology

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

extracted_records = []

print("Starting extraction with multi-stage reasoning...\n")
for idx, note in enumerate(synthetic_notes):
    print(f"Analyzing Note #{idx + 1}...")
    try:
        distill_prompt = (
            "Summarize the following clinical note into a concise medical profile. "
            "Focus ONLY on: Metadata (patient_id, document_reference_id, note_date, title), "
            "Patient demographics, Assessment and Plan (CRITICAL: preserve the exact phrase if admitted for 'epilepsy surgery evaluation'), "
            "Principal Diagnoses, hand dominance, employment, "
            "detailed seizure history (CRITICAL: MUST include exact seizure onset age, frequency, clusters, refractoriness, types, and awareness levels), "
            "past medical history, past surgical history, social history (MUST include driving status), "
            "ALL medications (including side effects), and ALL NEUROIMAGING/PREVIOUS WORKUP. "
            "IGNORE: Physical exam findings, vital signs, and current lab results."
        )
        distilled_summary = llm.invoke(f"{distill_prompt}\n\n{note}")
        print("\n--- PASS 1: DISTILLED SUMMARY ---")
        print(distilled_summary.content)

        # PASS 2: Parallel Extraction
        history_result = history_chain.invoke({"clinical_note": distilled_summary.content})
        history_data = history_result if isinstance(history_result, dict) else history_result.model_dump()
        
        meds_result = meds_chain.invoke({"clinical_note": distilled_summary.content})
        meds_data = meds_result if isinstance(meds_result, dict) else meds_result.model_dump()

        imaging_result = imaging_chain.invoke({"clinical_note": distilled_summary.content})
        imaging_data = imaging_result if isinstance(imaging_result, dict) else imaging_result.model_dump()

        data = {**history_data, **meds_data, **imaging_data}
        combined_reasoning = history_data.get("step_by_step_logic", []) + meds_data.get("step_by_step_logic", []) + imaging_data.get("step_by_step_logic", [])

        # --- THE BULLETPROOF SAFETY NET ---
        for step in combined_reasoning:
            var_name = step.get('variable_name')
            code_str = str(step.get('chosen_code', ''))
            
            if var_name and not data.get(var_name) and code_str:
                extracted_numbers = re.findall(r'\d+', code_str)
                if extracted_numbers:
                    if var_name in ['medhx_etio_focal', 'medhx_priorepisgy_type', 'medhx_neurohx', 'medhx_psych', 'emu_asm_type', 'emu_asmdc_type']:
                        data[var_name] = [int(num) for num in extracted_numbers]
                    else:
                        data[var_name] = int(extracted_numbers[0])
        
        print("\n--- PASS 2: AI REASONING ---")
        for step in combined_reasoning:
            print(step)
        print("\n------------------------------\n")

        data['record_id'] = idx + 1 
        extracted_records.append(data)
        
    except Exception as e:
        print(f"Error processing Note #{idx + 1}: {e}")

# --- 5. EXPORT TO CSV ---
if extracted_records:
    df = pd.DataFrame(extracted_records)
    
    # Optimized checkbox expansion to fix Pandas Fragmentation warning
    def expand_checkboxes(dataframe, column_name, possible_codes):
        new_cols = {}
        for code in possible_codes:
            new_cols[f"{column_name}___{code}"] = dataframe[column_name].apply(
                lambda x: 1 if (
                    (isinstance(x, list) and code in x) or 
                    (isinstance(x, (int, float)) and x == code) or
                    (isinstance(x, str) and str(code) in x)
                ) else 0
            )
        new_df = pd.DataFrame(new_cols)
        return pd.concat([dataframe.drop(columns=[column_name]), new_df], axis=1)

    if 'medhx_priorepisgy_type' in df.columns:
        df = expand_checkboxes(df, 'medhx_priorepisgy_type', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 999])
    if 'medhx_neurohx' in df.columns:
        df = expand_checkboxes(df, 'medhx_neurohx', [1, 2, 3, 4, 5, 6, 7, 8, 0, 999])
    if 'medhx_etio_focal' in df.columns:
        df = expand_checkboxes(df, 'medhx_etio_focal', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999])
    if 'medhx_psych' in df.columns:
        df = expand_checkboxes(df, 'medhx_psych', [1, 2, 3, 4, 5, 6, 7, 0, 999])

    asm_codes = list(range(1, 30)) + [99]
    if 'emu_asm_type' in df.columns:
        df = expand_checkboxes(df, 'emu_asm_type', asm_codes)
    if 'emu_asmdc_type' in df.columns:
        df = expand_checkboxes(df, 'emu_asmdc_type', asm_codes)

    if 'internal_clinical_reasoning' in df.columns:
        df = df.drop(columns=['internal_clinical_reasoning'])
    if 'step_by_step_logic' in df.columns:
        df = df.drop(columns=['step_by_step_logic'])
    
    cols = ['record_id'] + [col for col in df.columns if col != 'record_id']
    df = df[cols]
    
    export_filename = "redcap_import_ready.csv"
    df.to_csv(export_filename, index=False)
    print(f"\nExtraction complete! Saved {len(df)} records to {export_filename}")