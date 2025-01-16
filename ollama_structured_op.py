from ollama import chat
from pydantic import BaseModel, Field
from typing import List, Literal, Union
from data.notes import notes
import csv
from datetime import datetime
import os
from data_fields import PrimaryDiagnosis, HospitalCourse, LabResults, Medications
import argparse
from utils import parse_clinical_note

class ClinicalExtract(BaseModel):
    primary_diagnosis: PrimaryDiagnosis
    hospital_course: HospitalCourse
    lab_data: LabResults
    medications: Medications

AVAILABLE_MODELS = {
    'llama3.3': 'llama3.3',
    'llama3.3-q8': 'llama3.3:70b-instruct-q8_0',
    'llama3.1': 'llama3.1'
}

DEFAULT_MODEL = 'llama3.3'

def parse_args():
    parser = argparse.ArgumentParser(description='Clinical Note Information Extraction')
    parser.add_argument('--method', type=int, choices=[1, 2], default=2,
                       help='Note feeding method: 1=whole note, 2=section by section')
    parser.add_argument('--model', type=str, choices=list(AVAILABLE_MODELS.keys()), 
                       default=DEFAULT_MODEL,
                       help=f'LLM model to use. Available models: {", ".join(AVAILABLE_MODELS.keys())}')
    args = parser.parse_args()
    
    args.model = AVAILABLE_MODELS[args.model]
    return args

def extract_whole_note(notes, llm_model):
    results = []
    for note_key in notes.keys():
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': f"extract information from the following clinical note: {notes[note_key]}",
                }
            ],
            model=llm_model,
            format=ClinicalExtract.model_json_schema(),
        )

        clinical_extract = ClinicalExtract.model_validate_json(response.message.content)
        
        flat_data = {
            'note_id': note_key,
            'primary_disease': clinical_extract.primary_diagnosis.primary_disease,
            'conditioning_regimen': clinical_extract.primary_diagnosis.conditioing_regimen,
            'donor_type': clinical_extract.primary_diagnosis.donor_type,
            'transplant_complications': ','.join(clinical_extract.primary_diagnosis.transplant_related_complications),
            'reason_for_admission': clinical_extract.hospital_course.reason_for_admission,
            'problem_list': ','.join(clinical_extract.hospital_course.problem_list),
            'wbc_admission': clinical_extract.lab_data.wbc_admission,
            'wbc_discharge': clinical_extract.lab_data.wbc_discharge,
            'neuts_admission': clinical_extract.lab_data.neuts_admission,
            'neuts_discharge': clinical_extract.lab_data.neuts_discharge,
            'hgb_admission': clinical_extract.lab_data.hgb_admission,
            'hgb_discharge': clinical_extract.lab_data.hgb_discharge,
            'plt_admission': clinical_extract.lab_data.plt_admission,
            'plt_discharge': clinical_extract.lab_data.plt_discharge,
            't_bili_admission': clinical_extract.lab_data.t_bili_admission,
            't_bili_discharge': clinical_extract.lab_data.t_bili_discharge,
            'ca_admission': clinical_extract.lab_data.ca_admission,
            'ca_discharge': clinical_extract.lab_data.ca_discharge,
            'medications_admission': ','.join(clinical_extract.medications.medications_admission),
            'medications_discharge': ','.join(clinical_extract.medications.medications_discharge)
        }
        
        results.append(flat_data)
        print(f"\nResults for {note_key}:")
        print(clinical_extract)
    
    return results


def extract_section_by_section(notes, llm_model):
    results = []
    
    for note_key in notes.keys():
        sections = parse_clinical_note(notes[note_key])
        
        # Extract specific fields from relevant sections
        primary_diagnosis_prompt = f"""
        Extract primary diagnosis information from these sections:
        History of Present Illness: {sections['History of Present Illness']}
        Discharge Diagnosis: {sections['Discharge Diagnosis']}
        """

        transplant_info_prompt = f"""
        Extract transplant information from these sections:
        Active Issues: {sections['Active Issues']}
        Discharge Diagnosis: {sections['Discharge Diagnosis']}
        """

        hospital_course_prompt = f"""
        Extract hospital course information from these sections:
        Discharge Diagnosis: {sections['Discharge Diagnosis']}
        Active Issues: {sections['Active Issues']}
        """
        
        lab_results_prompt = f"""
        Extract lab results from this section:
        Pertinent Results: {sections['Pertinent Results']}
        """
        
        medications_prompt = f"""
        Extract medications from these sections:
        Medications on Admission: {sections['Medications on Admission']}
        Discharge Medications: {sections['Discharge Medications']}
        """
        
        # Get responses from LLM for each section
        diagnosis_response = chat(
            messages=[{'role': 'user', 'content': primary_diagnosis_prompt}],
            model=llm_model,
            format=PrimaryDiagnosis.model_json_schema(),
        )

        transplant_info_response = chat(
            messages=[{'role': 'user', 'content': transplant_info_prompt}],
            model=llm_model,
            format=PrimaryDiagnosis.model_json_schema(),
        )
        
        hospital_course_response = chat(
            messages=[{'role': 'user', 'content': hospital_course_prompt}],
            model=llm_model,
            format=HospitalCourse.model_json_schema(),
        )
        
        labs_response = chat(
            messages=[{'role': 'user', 'content': lab_results_prompt}],
            model=llm_model,
            format=LabResults.model_json_schema(),
        )
        
        medications_response = chat(
            messages=[{'role': 'user', 'content': medications_prompt}],
            model=llm_model,
            format=Medications.model_json_schema(),
        )
        
        # Parse responses
        diagnosis_data = PrimaryDiagnosis.model_validate_json(diagnosis_response.message.content)
        transplant_info_data = PrimaryDiagnosis.model_validate_json(transplant_info_response.message.content)
        hospital_course_data = HospitalCourse.model_validate_json(hospital_course_response.message.content)
        lab_data = LabResults.model_validate_json(labs_response.message.content)
        medications_data = Medications.model_validate_json(medications_response.message.content)
        
        # Create flat data structure
        flat_data = {
            'note_id': note_key,
            'primary_disease': diagnosis_data.primary_disease,
            'conditioning_regimen': transplant_info_data.conditioing_regimen,
            'donor_type': transplant_info_data.donor_type,
            'transplant_complications': ','.join(transplant_info_data.transplant_related_complications),
            'reason_for_admission': hospital_course_data.reason_for_admission,
            'problem_list': ','.join(hospital_course_data.problem_list),
            'wbc_admission': lab_data.wbc_admission,
            'wbc_discharge': lab_data.wbc_discharge,
            'neuts_admission': lab_data.neuts_admission,
            'neuts_discharge': lab_data.neuts_discharge,
            'hgb_admission': lab_data.hgb_admission,
            'hgb_discharge': lab_data.hgb_discharge,
            'plt_admission': lab_data.plt_admission,
            'plt_discharge': lab_data.plt_discharge,
            't_bili_admission': lab_data.t_bili_admission,
            't_bili_discharge': lab_data.t_bili_discharge,
            'ca_admission': lab_data.ca_admission,
            'ca_discharge': lab_data.ca_discharge,
            'medications_admission': ','.join(medications_data.medications_admission),
            'medications_discharge': ','.join(medications_data.medications_discharge)
        }
        
        results.append(flat_data)
        print(f"\nResults for {note_key}:")
        print(flat_data)
    
    return results


def save_results(results, llm_model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'notes_extracts_{llm_model}_{timestamp}.csv'
    
    os.makedirs('results', exist_ok=True)
    
    with open(f'results/{filename}', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to results/{filename}")

def main():
    args = parse_args()
    
    if args.method == 1:
        results = extract_whole_note(notes, args.model)
    else:
        results = extract_section_by_section(notes, args.model)
    
    save_results(results, args.model)

if __name__ == "__main__":
    main()
