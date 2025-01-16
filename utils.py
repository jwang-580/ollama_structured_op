import re
from data.notes import notes

def parse_clinical_note(note_text):
    sections = {
        "History of Present Illness": "",
        "Past Medical History": "",
        "Physical Exam": "",
        "Pertinent Results": "",
        "Active Issues": "",
        "Medications on Admission": "",
        "Discharge Medications": "",
        "Discharge Diagnosis": "",
        "Discharge Condition": ""
    }
    
    # Make section headers case-insensitive and handle variations
    section_patterns = {
        re.compile(r'(?i)^\s*history\s+of\s+present\s+illness\s*:?\s*$'): "History of Present Illness",
        re.compile(r'(?i)^\s*past\s+medical\s+history\s*:?\s*$'): "Past Medical History",
        re.compile(r'(?i)^\s*physical\s+exam(?:ination)?\s*:?\s*$'): "Physical Exam",
        re.compile(r'(?i)^\s*pertinent\s+results\s*:?\s*$'): "Pertinent Results",
        re.compile(r'(?i)^\s*medications?\s+on\s+admission\s*:?\s*$'): "Medications on Admission",
        re.compile(r'(?i)^\s*discharge\s+medications?\s*:?\s*$'): "Discharge Medications",
        re.compile(r'(?i)^\s*discharge\s+diagnosis\s*:?\s*$'): "Discharge Diagnosis",
        re.compile(r'(?i)^\s*discharge\s+condition\s*:?\s*$'): "Discharge Condition",
        re.compile(r'(?i)^\s*active\s+issues\s*:?\s*$'): "Active Issues"
    }
    
    current_section = None
    section_content = []
    
    for line in note_text.split('\n'):
        line = line.strip()
        
        # Check if line matches any section header pattern
        matched = False
        for pattern, section_name in section_patterns.items():
            if pattern.match(line):
                if current_section:
                    sections[current_section] = '\n'.join(section_content).strip()
                current_section = section_name
                section_content = []
                matched = True
                break
                
        if not matched and current_section and line:
            section_content.append(line)
    
    # Save the last section
    if current_section and section_content:
        sections[current_section] = '\n'.join(section_content).strip()
    
    return sections

if __name__ == "__main__":
    note_text = notes['note_1']
    sections = parse_clinical_note(note_text)
    print(sections)