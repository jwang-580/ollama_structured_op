# using GPT-4o as ground truth
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Literal, Union
from data.notes import notes
from data_fields import PrimaryDiagnosis, HospitalCourse, LabResults, Medications
from datetime import datetime
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv('OAI_API_KEY'))

class ClinicalExtract(BaseModel):
    primary_diagnosis: PrimaryDiagnosis
    hospital_course: HospitalCourse
    lab_data: LabResults
    medications: Medications

results = []

for note_key in notes.keys():
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "user", "content": f"extract information from the following clinical note: {notes[note_key]}"}
        ],
        response_format=ClinicalExtract
    )
    clinical_extract = response.choices[0].message.parsed

    flat_data = {
        'note_id': note_key,
        'primary_disease': clinical_extract.primary_diagnosis.primary_disease,
        'conditioning_regimen_type': clinical_extract.primary_diagnosis.conditioning_regimen_type,
        'conditioning_regimen': clinical_extract.primary_diagnosis.conditioning_regimen,
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'notes_extracts_gpt4o_{timestamp}.csv'

os.makedirs('results', exist_ok=True)

with open(f'results/{filename}', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to results/{filename}")