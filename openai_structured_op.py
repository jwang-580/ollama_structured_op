# using GPT-4o as ground truth
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import List, Literal, Union

load_dotenv()

client = OpenAI(api_key=os.getenv('OAI_API_KEY'))

class PrimaryDiagnosis(BaseModel):
    primary_disease: str = Field(..., description="The primary disease indicated for bone marrow transplantation")
    conditioing_regimen: str = Field(..., description="The conditioning regimen used for bone marrow transplant") 
    donor_type: list[Literal["Unrelated matched donor", "Unrelated mismatched donor", "Related matched donor", "Haploidentical donor", "Not mentioned"]] = Field(
        ..., 
        description="The type of donor used for bone marrow transplant"
    )
    transplant_related_complications: list[str] = Field(..., description="Complications specifically related to bone marrow transplant")

class HospitalCourse(BaseModel):
    reason_for_admission: list[Literal["Infection", "GVHD", "Respiratory failure", "Disease relapse", "Bone marrow trasnpalnt", "Other"]] = Field(..., description="The reason for this admission to the hospital")
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