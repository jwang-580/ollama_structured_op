from pydantic import BaseModel, Field
from typing import List, Literal, Union

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
    wbc_admission: float = Field(..., description="white blood cell count at admission, return 999 if not mentioned")
    wbc_discharge: float = Field(..., description="white blood cell count at discharge, return 999 if not mentioned")
    neuts_admission: float = Field(..., description="neutrophil count at admission, return 999 if not mentioned")
    neuts_discharge: float = Field(..., description="neutrophil count at discharge, return 999 if not mentioned")
    hgb_admission: float = Field(..., description="hemoglobin level at admission, return 999 if not mentioned")
    hgb_discharge: float = Field(..., description="hemoglobin level at discharge, return 999 if not mentioned")
    plt_admission: float = Field(..., description="platelet count at admission, return 999 if not mentioned")
    plt_discharge: float = Field(..., description="platelet count at discharge, return 999 if not mentioned")
    t_bili_admission: float = Field(..., description="total bilirubin level at admission, return 999 if not mentioned")
    t_bili_discharge: float = Field(..., description="total bilirubin level at discharge, return 999 if not mentioned")
    ca_admission: float = Field(..., description="calcium level at admission, return 999 if not mentioned")
    ca_discharge: float = Field(..., description="calcium level at discharge, return 999 if not mentioned")

class Medications(BaseModel):
    medications_admission: list[str] = Field(..., description="all medication names at admission")
    medications_discharge: list[str] = Field(..., description="all medication names at discharge")
