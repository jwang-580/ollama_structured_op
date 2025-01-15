from ollama import chat
from pydantic import BaseModel, Field
from typing import List, Literal, Union
from data.notes import notes
import csv
from datetime import datetime
import os

class PrimaryDiagnosis(BaseModel):
    primary_disease: str = Field(..., description="The primary disease indicated for bone marrow transplantation")
    conditioing_regimen: str = Field(..., description="The conditioning regimen used for bone marrow transplant") 
    donor_type: Literal["Unrelated matched donor", "Unrelated mismatched donor", "Related matched donor", "Haploidentical donor", "Not mentioned"] = Field(
        ..., 
        description="The type of donor used for bone marrow transplant"
    )
    transplant_related_complications: list[str] = Field(..., description="Complications specifically related to bone marrow transplant")

class HospitalCourse(BaseModel):
    reason_for_admission: Literal["Infection", "GVHD", "Respiratory failure", "Disease relapse", "Bone marrow trasnpalnt", "Other"] = Field(..., description="The reason for this admission to the hospital")
    problem_list: list[str] = Field(..., description="Problem list during hospital stay, be sccucinct")

class LabResults(BaseModel):
    wbc_admission: float = Field(..., description="white blood cell count at admission")
    wbc_discharge: float = Field(..., description="white blood cell count at discharge")
    neuts_admission: float = Field(..., description="neutrophil count at admission")
    neuts_discharge: float = Field(..., description="neutrophil count at discharge")
    hgb_admission: float = Field(..., description="hemoglobin level at admission")
    hgb_discharge: float = Field(..., description="hemoglobin level at discharge")
    plt_admission: float = Field(..., description="platelet count at admission")
    plt_discharge: float = Field(..., description="platelet count at discharge")
    t_bili_admission: float = Field(..., description="total bilirubin level at admission")
    t_bili_discharge: float = Field(..., description="total bilirubin level at discharge")
    ca_admission: float = Field(..., description="calcium level at admission")
    ca_discharge: float = Field(..., description="calcium level at discharge")

class Medications(BaseModel):
    medications_admission: list[str] = Field(..., description="medications at admission")
    medications_discharge: list[str] = Field(..., description="medications at discharge")

class ClinicalExtract(BaseModel):
    primary_diagnosis: PrimaryDiagnosis
    hospital_course: HospitalCourse
    lab_data: LabResults
    medications: Medications

results = []
llm_model = 'llama3.3'
# llm_model='llama3.3:70b-instruct-q8_0'

for note_key in notes.keys():
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': f"extract inforamtion from the following clinical note: {notes[note_key]}",
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'notes_extracts_{llm_model}_{timestamp}.csv'

os.makedirs('results', exist_ok=True)

with open(f'results/{filename}', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to results/{filename}")
