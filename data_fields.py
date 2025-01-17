from pydantic import BaseModel, Field
from typing import List, Literal, Union

class PrimaryDiagnosis(BaseModel):
    primary_disease: str = Field(..., description="The primary disease indicated for bone marrow transplantation with full name: e.g. Acute Myeloid Leukemia, do not use abbreviation")
    conditioning_regimen_type: list[Literal["Reduced intensity conditioning", "Myeloablative conditioning"]] = Field(..., description="The type of conditioning regimen used for bone marrow transplant")
    conditioning_regimen: list[str] = Field(..., description="The conditioning regimen used for bone marrow transplant with full name: e.g. Fludarabine, Busulfan") 
    donor_type: list[Literal["Unrelated matched donor", "Unrelated mismatched donor", "Related matched donor", "Haploidentical donor", "Not mentioned"]] = Field(
        ..., 
        description="The type of donor used for bone marrow transplant"
    )
    transplant_related_complications: list[str] = Field(..., description="Complications specifically related to bone marrow transplant")

class HospitalCourse(BaseModel):
    reason_for_admission: list[Literal["Infection", "GVHD", "Respiratory failure", "Disease relapse", "Bone marrow trasnpalnt", "Other"]] = Field(..., description="The reason for this admission to the hospital")
    problem_list: list[str] = Field(..., description="Problem list during hospital stay, be sccucinct")

class LabResults(BaseModel):
    wbc_admission: float = Field(..., description="Provide the white blood cell count at admission. If the information is not mentioned, return the value 999.")
    wbc_discharge: float = Field(..., description="Provide the white blood cell count at discharge. If the information is not mentioned, return the value 999.")
    neuts_admission: float = Field(..., description="Provide the neutrophil count at admission. If the information is not mentioned, return the value 999.")
    neuts_discharge: float = Field(..., description="Provide the neutrophil count at discharge. If the information is not mentioned, return the value 999.")
    hgb_admission: float = Field(..., description="Provide the hemoglobin level at admission. If the information is not mentioned, return the value 999.")
    hgb_discharge: float = Field(..., description="Provide the hemoglobin level at discharge. If the information is not mentioned, return the value 999.")
    plt_admission: float = Field(..., description="Provide the platelet count at admission. If the information is not mentioned, return the value 999.")
    plt_discharge: float = Field(..., description="Provide the platelet count at discharge. If the information is not mentioned, return the value 999.")
    t_bili_admission: float = Field(..., description="Provide the total bilirubin level at admission. If the information is not mentioned, return the value 999.")
    t_bili_discharge: float = Field(..., description="Provide the total bilirubin level at discharge. If the information is not mentioned, return the value 999.")
    ca_admission: float = Field(..., description="Provide the calcium level at admission. If the information is not mentioned, return the value 999.")
    ca_discharge: float = Field(..., description="Provide the calcium level at discharge. If the information is not mentioned, return the value 999.")

class Medications(BaseModel):
    medications_admission: list[str] = Field(..., description="all medication names at admission, do not include the dosage")
    medications_discharge: list[str] = Field(..., description="all medication names at discharge, do not include the dosage")
