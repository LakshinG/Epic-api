import json
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import re

class VariableReasoning(BaseModel):
    variable_name: str = Field(description="The name of the REDCap field (e.g., medhx_neurohx)")
    evidence_quote: str = Field(description="Exact sentence from the summary proving your choice.")
    chosen_code: str = Field(description="DO NOT WRITE SENTENCES. ONLY OUTPUT DIGITS (e.g. '1' or '63'). No words, no letters. If multiple, list them separated by commas (e.g., '1, 4'). If empty or not applicable, write 'NONE'.")

# 1. Pydantic Schema with Schema-Bound Constraints
class REDCapEpilepsyData(BaseModel):
    step_by_step_logic: List[VariableReasoning] = Field(
        description="MANDATORY: You must create a reasoning entry for EVERY REDCap variable (sz_age, hand_dom, medhx_etio, medhx_prior_episgy, demo_gender, demo_employed, medhx_szsyndrome, medhx_szsyndrome_type, medhx_etio_focal, medhx_priorepisgy_type, medhx_neurohx, medhx_psych, medhx_si, medhx_driving, medhx_sgy_cand_yn, emu_sz_type, emu_sz_type1_freq, emu_asm_number, emu_asm_type, emu_asm_sfx, emu_asmdc_number, emu_asmdc_type, emu_dcevents_type, emu_epilepsytype, emu_epilepsy_intract, emu_sxcandidate, mri_yn, mri_normal_abnormal, mri_lateralization, mri_l_localization, mri_r_localization, mri_lesion_left, mri_lesion_right, pet_yn, fmri_yn, wada_yn). Cite the text and justify the code BEFORE assigning the final variables."
    )
    sz_age: Optional[int]
    hand_dom: Optional[int]
    medhx_etio: Optional[int]
    medhx_prior_episgy: Optional[int]
    demo_gender: Optional[int]
    demo_employed: Optional[int]
    medhx_szsyndrome: Optional[int]
    medhx_szsyndrome_type: Optional[int]
    medhx_si: Optional[int]
    medhx_driving: Optional[int]
    medhx_sgy_cand_yn: Optional[int]
    emu_sz_type: Optional[int]
    emu_sz_type1_freq: Optional[int]
    emu_asm_number: Optional[int]
    emu_asm_sfx: Optional[int]
    emu_asmdc_number: Optional[int]
    emu_dcevents_type: Optional[int]
    emu_epilepsytype: Optional[int]
    emu_epilepsy_intract: Optional[int]
    emu_sxcandidate: Optional[int]
    mri_yn: Optional[int]
    mri_normal_abnormal: Optional[int]
    mri_lateralization: Optional[int]
    mri_l_localization: Optional[int]
    mri_r_localization: Optional[int]
    mri_lesion_left: Optional[int]
    mri_lesion_right: Optional[int]
    pet_yn: Optional[int]
    fmri_yn: Optional[int]
    wada_yn: Optional[int]
    # --- MULTI-SELECT FIELDS ---
    medhx_etio_focal: Optional[List[int]]
    medhx_priorepisgy_type: Optional[List[int]]
    medhx_neurohx: Optional[List[int]]
    medhx_psych: Optional[List[int]]
    emu_asm_type: Optional[List[int]]
    emu_asmdc_type: Optional[List[int]]

# 2. Initialize the 14B Model
llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)
structured_llm = llm.with_structured_output(REDCapEpilepsyData)

# 3. Explicit System Instructions (Mapping Rules)
system_instructions = """
You are an expert clinical data abstraction AI. 
TASK: Extract REDCap variables from the clinical note into specific integer codes.

UNIVERSAL VERIFICATION RULES:
1. REASONING FIRST: You must evaluate EVERY single field in the `step_by_step_logic` list before outputting any final numbers. For each field, provide the variable name, cite the exact evidence, and state the chosen code. DO NOT write sentences in the chosen_code field; ONLY output the final mapped integer digit(s).
2. NEGATION CHECK: If a sentence contains "no history of", "denies", "negative for", or "not present", you MUST map that field to 0 or null.
3. CONTEXT CHECK: Ensure the diagnosis refers to the PATIENT, not family members.

CODE MAPPINGS (STRICT PDF VERIFICATION):
- sz_age: The patient's age at FIRST seizure onset.
- hand_dom: 1=Left, 2=Right, 3=Ambidextrous, 99=Other.
- medhx_etio: Seizure type. 0=Generalized, 1=Focal/Multifocal, 2=Both, 3=Psychogenic, 4=Physiologic.
- medhx_prior_episgy: PREVIOUS EPILEPSY SURGERY? 1=Yes, 2=No.
- demo_gender: 1=Male, 2=Female, 3=Transgender, 4=Non-binary, 99=Other.
- demo_employed: 1=Yes, 0=No, 999=Unknown.
- medhx_szsyndrome: Confirmed epilepsy syndrome presence. 1=Yes, 2=No. (NOTE: "Localization-related epilepsy" is NOT a named syndrome, map to 2).
- medhx_szsyndrome_type: Syndrome name. 1=MTLE-HS, 2=LGS, 4=CAE, 7=Dravet, 12=JME, 13=Focal/Multifocal NOS, 14=JAE, 15=Genetic/Idiopathic NOS, 999=Other.
- medhx_etio_focal: Specific structural cause of Focal Seizures. 1=Mesial-temporal sclerosis, 2=Prior TBI, 3=Post-stroke/Vascular injury, 4=Post-infectious, 5=Tumor, 6=Vascular lesion, 7=Cortical Dysplasia, 8=Autoimmune, 9=Genetic, 10=Other Lesion, 999=Unknown.
- medhx_priorepisgy_type: Prior epilepsy surgeries. 10=Multiple subpial transections, 11=VNS, 12=DBS, 13=RNS, 14=Other, 999=Unknown.
- medhx_neurohx: 1=Stroke, 2=Hemorrhage, 3=TBI, 4=Dementia, 5=Headaches/Neuropathy, 0=None.
- medhx_psych: 1=Depression, 2=Anxiety, 3=Bipolar Disorder, 4=PTSD, 5=Schizophrenia, 6=Alcohol/Substance Use, 7=Other, 0=None, 999=Unknown.
- medhx_si: Suicidal Ideation/Attempt? 1=Yes, 0=No.
- medhx_driving: Currently driving? 1=Yes, 2=No, 3=Unclear/Unknown.
- medhx_sgy_cand_yn: Surgical candidate? 1=Yes, 2=No, 3=Unclear.
- emu_sz_type: Most frequent seizure type. 1=Generalized TC, 2=Focal motor aware, 3=Focal non-motor aware, 4=Focal motor impaired, 5=Focal non-motor impaired, 6=Aware NOS, 7=Staring spells NOS, 8=Hypermotor NOS, 9=Myoclonus, 10=Convulsions NOS, 99=Other.
- emu_sz_type1_freq: Frequency. 1=Multiple/day, 2=Daily, 3=Multiple/week, 4=Weekly, 5=Multiple/month, 6=Monthly, 7=Multiple/year, 8=Yearly, 9=Random clusters, 99=Other.
- emu_asm_number: Number of ASMs on Admission. 0=None, 1=One, 2=Two, 3=Three, 4=Four, 5=Five+.
- emu_asm_type: ASM list. 1=levetiracetam, 2=lamotrigine, 3=carbamazepine, 4=oxcarbazepine, 6=brivaracetam, 7=topiramate, 8=zonisamide, 9=clobazam, 10=clonazepam, 12=lorazepam, 13=valproic acid, 14=gabapentin, 15=lacosamide, 16=pregabalin, 17=phenytoin, 24=perampanel, 99=Other.
- emu_asm_sfx: Side effects from ASMs? 1=yes, 2=no, 99=unclear.
- emu_asmdc_number: Number of ASMs on Discharge. 0=None, 1=One, 2=Two, 3=Three, 4=Four, 5=Five+.
- emu_asmdc_type: ASM list on discharge. Same codes as emu_asm_type.
- emu_dcevents_type: Discharge diagnosis. 1=Epilepsy, 2=FND, 3=Mixed FND/Epilepsy, 4=Physiologic Non-epileptic, 5=Inconclusive.
- emu_epilepsytype: 1=Focal Single, 2=Focal Two foci, 3=Multifocal, 4=Generalized Idiopathic, 5=Generalized Symptomatic, 6=Unlocalizable.
- emu_epilepsy_intract: Medically refractory? 1=Yes, 2=No, 3=Unclear.
- emu_sxcandidate: Surgery Candidate (EMU)? 1=Yes/discussed, 2=Yes/not amenable, 3=Yes/not discussed, 4=Possible future, 5=No, 999=Unknown.
- mri_yn: MRI performed? 1=Yes, 2=No but ordered, 0=No.
- mri_normal_abnormal: MRI Normal? 1=Normal, 2=Abnormal.
- mri_lateralization: 1=Left, 2=Right, 3=Bilateral, 4=Midline, 5=Multifocal.
- mri_l_localization: Left MRI. 1=Temporal, 2=Frontal, 3=Parietal, 4=Occipital, 5=Sub-cortical, 6=Other.
- mri_r_localization: Right MRI. 1=Temporal, 2=Frontal, 3=Parietal, 4=Occipital, 5=Sub-cortical, 6=Other.
- mri_lesion_left: Left lesion. 1=Hippocampal sclerosis, 2=Cavernoma, 3=Neoplasm/DNET/glioma, 4=FCD, 5=Stroke/Encephalomalacia, 6=Polymicrogyria, 7=Heterotopias, 8=Cortical Tubers, 0=Multiple, 99=Other.
- mri_lesion_right: Right lesion. Same codes as left.
- pet_yn: FDG-PET? 1=Yes, 2=No but ordered, 0=No.
- fmri_yn: fMRI? 1=Yes, 2=No but ordered, 0=No.
- wada_yn: WADA? 1=Yes, 2=No but ordered, 0=No.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{clinical_note}")
])
extraction_chain = prompt | structured_llm

# 4. Data Source
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

# 5. The Generalized Two-Pass Processing Loop with Safety Net
extracted_records = []

print("Starting extraction with multi-stage reasoning...\n")
for idx, note in enumerate(synthetic_notes):
    print(f"Analyzing Note #{idx + 1}...")
    try:
        # PASS 2: Structured Extraction (Chain of Thought)
        data = extraction_chain.invoke({"clinical_note": note})
        
        if not isinstance(data, dict):
            data = data.model_dump()

        # --- THE BULLETPROOF SAFETY NET ---
        reasoning_list = data.get("step_by_step_logic", [])
        for step in reasoning_list:
            var_name = step.get('variable_name')
            code_str = str(step.get('chosen_code', ''))
            
            # If the main variable is empty but the reasoning has an answer, extract it!
            if var_name and not data.get(var_name) and code_str:

                # Extract ONLY the numbers from strings like "999=Unknown" or "1=Depression, 4=PTSD"
                extracted_numbers = re.findall(r'\d+', code_str)

                if extracted_numbers:
                    # Check if Pydantic expects a list for this specific variable
                    if var_name in ['medhx_etio_focal', 'medhx_priorepisgy_type', 'medhx_neurohx', 'medhx_psych', 'emu_asm_type', 'emu_asmdc_type']:
                        data[var_name] = [int(num) for num in extracted_numbers]
                    else:
                        data[var_name] = int(extracted_numbers[0]) # Grab the first number found

        print("\n--- PASS 2: AI REASONING ---")
        if reasoning_list:
            for step in reasoning_list:
                print(step)
        else:
            print("No reasoning provided.")
        print("\n------------------------------\n")

        data['record_id'] = idx + 1 
        extracted_records.append(data)
        
    except Exception as e:
        print(f"Error processing Note #{idx + 1}: {e}")

# 6. Export to CSV using Pandas
if extracted_records:
    df = pd.DataFrame(extracted_records)
    
    # REDCap Checkbox Expander
    # NEW: Bulletproof Checkbox Expander
    def expand_checkboxes(dataframe, column_name, possible_codes):
        for code in possible_codes:
            dataframe[f"{column_name}___{code}"] = dataframe[column_name].apply(
                lambda x: 1 if (
                    (isinstance(x, list) and code in x) or
                    (isinstance(x, (int, float)) and x == code) or
                    (isinstance(x, str) and str(code) in x)
                ) else 0
            )
        return dataframe.drop(columns=[column_name])

    # Expand multi-select columns with verified PDF codes
    if 'medhx_priorepisgy_type' in df.columns:
        df = expand_checkboxes(df, 'medhx_priorepisgy_type', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 999])
        
    if 'medhx_neurohx' in df.columns:
        df = expand_checkboxes(df, 'medhx_neurohx', [1, 2, 3, 4, 5, 6, 7, 8, 0, 999])

    if 'medhx_etio_focal' in df.columns:
        df = expand_checkboxes(df, 'medhx_etio_focal', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999])

    if 'medhx_psych' in df.columns:
        df = expand_checkboxes(df, 'medhx_psych', [1, 2, 3, 4, 5, 6, 7, 0, 999])

    if 'emu_asm_type' in df.columns:
        df = expand_checkboxes(df, 'emu_asm_type', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 99])

    if 'emu_asmdc_type' in df.columns:
        df = expand_checkboxes(df, 'emu_asmdc_type', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 99])

    # Clean the dataframe for REDCap import
    if 'internal_clinical_reasoning' in df.columns:
        df = df.drop(columns=['internal_clinical_reasoning'])

    # Drop the reasoning column before export
    if 'step_by_step_logic' in df.columns:
        df = df.drop(columns=['step_by_step_logic'])
    
    # Reorder columns so record_id is first
    cols = ['record_id'] + [col for col in df.columns if col != 'record_id']
    df = df[cols]
    
    export_filename = "redcap_import_ready.csv"
    df.to_csv(export_filename, index=False)
    print(f"\nExtraction complete! Saved {len(df)} records to {export_filename}")